# src/workflows/user_workflow.py
from typing import Any, Dict, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from src.api.battles import get_player_battlelog
from src.analytics.battle_filters import filter_and_normalize_ranked_1v1
from src.analytics.user_analytics import compute_user_analytics
from src.analytics.plots import generate_card_plots

# ---------- LangGraph State Definition ----------


class UserAnalyticsState(TypedDict, total=False):
    """
    State for the Phase 1 user analytics workflow.

    Fields:

      - player_tag: input from user (e.g. "#8C8JJQLG")
      - battles_raw: raw battlelog list from Clash Royale API
      - battles_filtered: ranked/Trophy Road 1v1 battles (normalized)
      - user_analytics: analytics dict (same schema as meta_analytics)
      - user_plots: optional dict of plot paths (usually analytics["plots"])
      - notes: optional list of debug / info strings
    """

    player_tag: str
    battles_raw: List[Dict[str, Any]]
    battles_filtered: List[Dict[str, Any]]
    user_analytics: Dict[str, Any]
    user_llm_tables: Dict[str, Any]
    user_plots: Dict[str, Any]
    notes: List[str]
    
# ---------- LLM helper table builders ----------

def build_user_summary_table(summary: dict):
    table = []
    for key, value in (summary or {}).items():
        table.append({"metric": key, "value": value})
    return table


def build_user_deck_summary(user_analytics):
    """
    Build a per-deck-type summary for the user from user_analytics["my_deck_types"].

    We normalize this to a list of rows with a stable "deck_type" field.
    """
    table: List[dict] = []

    raw = user_analytics.get("my_deck_types") or []
    if not isinstance(raw, list):
        return table

    for row in raw:
        if not isinstance(row, dict):
            continue

        deck_type = (
            row.get("deck_type")
            or row.get("type")        
            or row.get("my_deck_type")
        )
        if not deck_type:
            continue

        new_row = dict(row)  # shallow copy
        new_row["deck_type"] = deck_type
        table.append(new_row)

    return table



def build_user_matchup_summary(user_analytics):
    """
    Build a deck-type vs deck-type matchup table for the LLM.

    Source: user_analytics["deck_type_matchups"], which has rows like:

        {
          "my_deck_type": "Beatdown",
          "opp_deck_type": "Bridge Spam",
          "games": 6,
          "wins": 1,
          "losses": 5,
          "draws": 0,
          "win_rate": 0.1666...
        }

    We forward these rows as-is so the expert LLM can answer questions like:
      - "What deck types do I struggle vs?"
      - "Which archetypes am I best into?"
    """
    raw = user_analytics.get("deck_type_matchups") or []
    if not isinstance(raw, list):
        return []

    table: List[dict] = []

    for row in raw:
        if not isinstance(row, dict):
            continue

        my_type = row.get("my_deck_type")
        opp_type = row.get("opp_deck_type")
        if not my_type or not opp_type:
            continue

        # shallow copy; no difficulty labels
        out_row = dict(row)
        table.append(out_row)

    return table


def build_user_card_summary(user_analytics):
    """
    Merge best_cards + worst_cards into one per-card summary.

    We keep the original stats and add:
      - card_name (alias of card)
      - role: "best" / "worst"
      - source: "best_cards" / "worst_cards"
    """
    table: List[dict] = []

    best_cards = user_analytics.get("best_cards") or []
    worst_cards = user_analytics.get("worst_cards") or []

    # Best cards
    for row in best_cards:
        if not isinstance(row, dict):
            continue
        card_name = row.get("card") or row.get("card_name")
        if not card_name:
            continue

        new_row = dict(row)
        new_row["card_name"] = card_name
        new_row["role"] = "best"
        new_row["source"] = "best_cards"
        table.append(new_row)

    # Worst cards
    for row in worst_cards:
        if not isinstance(row, dict):
            continue
        card_name = row.get("card") or row.get("card_name")
        if not card_name:
            continue

        new_row = dict(row)
        new_row["card_name"] = card_name
        new_row["role"] = "worst"
        new_row["source"] = "worst_cards"
        table.append(new_row)

    return table

def build_opponent_card_summary(user_analytics):
    """
    Merge tough_opp_cards + easy_opp_cards into one per-opponent-card summary.
    We keep the stats and add:
      - card_name (alias of card)
      - role: "tough" / "easy"
      - source: "tough_opp_cards" / "easy_opp_cards"
    """
    table: List[dict] = []

    tough_opp = user_analytics.get("tough_opp_cards") or []
    easy_opp = user_analytics.get("easy_opp_cards") or []

    # Cards you struggle against
    for row in tough_opp:
        if not isinstance(row, dict):
            continue
        card_name = row.get("card") or row.get("card_name")
        if not card_name:
            continue

        new_row = dict(row)
        new_row["card_name"] = card_name
        new_row["role"] = "tough"
        new_row["source"] = "tough_opp_cards"
        table.append(new_row)

    # Cards you handle well
    for row in easy_opp:
        if not isinstance(row, dict):
            continue
        card_name = row.get("card") or row.get("card_name")
        if not card_name:
            continue

        new_row = dict(row)
        new_row["card_name"] = card_name
        new_row["role"] = "easy"
        new_row["source"] = "easy_opp_cards"
        table.append(new_row)

    return table



# ---------- Node: fetch_battlelog ----------


def fetch_battlelog_node(state: UserAnalyticsState) -> Dict[str, Any]:
    """
    Node:
      - reads player_tag from state
      - calls Clash Royale API
      - writes battles_raw into state
    """
    player_tag = state.get("player_tag")
    if not player_tag:
        raise ValueError("player_tag is required in state for fetch_battlelog_node")

    raw_battles = get_player_battlelog(player_tag)

    note = f"Fetched {len(raw_battles)} battles for {player_tag}"
    existing_notes = state.get("notes", []) or []
    existing_notes.append(note)

    return {
        "battles_raw": raw_battles,
        "notes": existing_notes,
    }


# ---------- Node: filter_and_normalize ----------


def filter_and_normalize_node(state: UserAnalyticsState) -> Dict[str, Any]:
    """
    Node:
      - reads battles_raw
      - filters to ranked/Trophy Road 1v1
      - normalizes battles
      - writes battles_filtered into state
    """
    battles_raw = state.get("battles_raw", []) or []
    if not battles_raw:
        raise ValueError("battles_raw is required in state for filter_and_normalize_node")

    battles_filtered = filter_and_normalize_ranked_1v1(battles_raw)

    note = f"Filtered to {len(battles_filtered)} ranked/Trophy Road 1v1 battles"
    existing_notes = state.get("notes", []) or []
    existing_notes.append(note)

    return {
        "battles_filtered": battles_filtered,
        "notes": existing_notes,
    }


# ---------- Node: compute_user_analytics ----------


def compute_user_analytics_node(state: UserAnalyticsState) -> Dict[str, Any]:
    """
    Node:
      - reads battles_filtered
      - computes analytics dict
      - writes user_analytics into state
    """
    battles_filtered = state.get("battles_filtered", []) or []
    if not battles_filtered:
        raise ValueError(
            "battles_filtered is required in state for compute_user_analytics_node"
        )

    analytics = compute_user_analytics(battles_filtered)

    note = (
        f"Computed analytics on {analytics.get('summary', {}).get('games_played', 0)} "
        "ranked/Trophy Road 1v1 battles"
    )
    existing_notes = state.get("notes", []) or []
    existing_notes.append(note)

    return {
        "user_analytics": analytics,
        "notes": existing_notes,
    }


# ---------- Node: generate_user_plots ----------


def generate_user_plots_node(state: UserAnalyticsState) -> Dict[str, Any]:
    """
    Node:
      - reads user_analytics
      - generates card-level plots
      - writes user_plots and updated user_analytics into state
    """
    analytics = state.get("user_analytics") or {}
    if not analytics:
        raise ValueError(
            "user_analytics is required in state for generate_user_plots_node"
        )

    analytics_with_plots = generate_card_plots(analytics, prefix="user")

    note = "Generated user card-level plots"
    existing_notes = state.get("notes", []) or []
    existing_notes.append(note)

    return {
        "user_analytics": analytics_with_plots,
        "user_plots": analytics_with_plots.get("plots", {}),
        "notes": existing_notes,
    }

## ---------- Node: llm tables ----------

def build_user_llm_tables_node(state: UserAnalyticsState) -> Dict[str, Any]:
    """
    Node:
      - reads user_analytics
      - builds compact LLM-ready tables
      - writes user_llm_tables into state

    Layout for user_llm_tables:
      - user_summary
      - user_deck_summary
      - user_matchup_summary
      - user_card_summary
      - opponent_card_summary
    """
    user_analytics = state.get("user_analytics") or {}
    if not user_analytics:
        raise ValueError(
            "user_analytics is required in state for build_user_llm_tables_node"
        )

    user_llm_tables = {
        "user_summary": build_user_summary_table(user_analytics.get("summary") or {}),
        "user_deck_summary": build_user_deck_summary(user_analytics),
        "user_matchup_summary": build_user_matchup_summary(user_analytics),
        "user_card_summary": build_user_card_summary(user_analytics),
        "opponent_card_summary": build_opponent_card_summary(user_analytics),
    }

    existing_notes = state.get("notes", []) or []
    existing_notes.append(
        "build_user_llm_tables: built "
        f"{len(user_llm_tables['user_deck_summary'])} deck rows, "
        f"{len(user_llm_tables['user_matchup_summary'])} matchup rows, "
        f"{len(user_llm_tables['user_card_summary'])} user card rows, "
        f"{len(user_llm_tables['opponent_card_summary'])} opponent card rows."
    )

    return {
        "user_llm_tables": user_llm_tables,
        "notes": existing_notes,
    }



# ---------- Graph Builder ----------


def build_user_analytics_graph():
    """
    Build the Phase 1 LangGraph workflow.

    Pipeline (D4):

        fetch_battlelog
            -> filter_and_normalize
            -> compute_user_analytics
            -> generate_user_plots
            -> END
    """
    graph = StateGraph(UserAnalyticsState)

    # Nodes
    graph.add_node("fetch_battlelog", fetch_battlelog_node)
    graph.add_node("filter_and_normalize", filter_and_normalize_node)
    graph.add_node("compute_user_analytics", compute_user_analytics_node)
    graph.add_node("build_user_llm_tables", build_user_llm_tables_node)  
    graph.add_node("generate_user_plots", generate_user_plots_node)

    # Entry / edges
    graph.set_entry_point("fetch_battlelog")
    graph.add_edge("fetch_battlelog", "filter_and_normalize")
    graph.add_edge("filter_and_normalize", "compute_user_analytics")
    graph.add_edge("compute_user_analytics", "build_user_llm_tables")  
    graph.add_edge("build_user_llm_tables", "generate_user_plots")       
    graph.add_edge("generate_user_plots", END)


    return graph.compile()
