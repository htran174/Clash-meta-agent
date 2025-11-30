# src/workflows/meta_workflow.py

from __future__ import annotations

import random
from typing import Any, Dict, List, Set
from typing import TypedDict

import matplotlib
matplotlib.use("Agg") 
import os
import matplotlib.pyplot as plt

from langgraph.graph import StateGraph, END

from src.api.players import fetch_top_players
from src.api.battles import get_player_battlelog
from src.analytics.battle_filters import filter_and_normalize_ranked_1v1
from src.analytics.meta_analytics import compute_meta_analytics
from src.analytics.meta_standardize import build_standardized_meta_table
from src.analytics.plots import (
    plot_deck_type_pie,
    plot_deck_type_bar,
    PLOTS_DIR,
    _ensure_plots_dir,
)

from src.analytics.meta_llm_tables import (
    build_meta_deck_summary,
    build_meta_matchup_summary,
)



# ---------------------------------------------------------------------------
# Constants for Phase 0 stopping condition
# ---------------------------------------------------------------------------

MIN_TOTAL_BATTLES = 2000
MIN_GAMES_PER_TYPE = 200

# Internal names are lowercased for robustness
REQUIRED_DECK_TYPES_LOWER = [
    "siege",
    "bait",
    "cycle",
    "bridge spam",
    "beatdown",
]


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class MetaState(TypedDict, total=False):
    """
    State used by the Phase 0 meta-analytics LangGraph.
    """

    # Players
    top_players: List[Dict[str, Any]]          # full top-player list from API
    selected_players: List[Dict[str, Any]]     # batch currently being fetched
    used_player_indices: Set[int]              # indices into top_players already used
    fetched_player_tags: Set[str]              # tags we've fetched logs for

    # Battles
    meta_raw_battles: List[Dict[str, Any]]     # all normalized ranked 1v1 battles
    normalized_battles: List[Dict[str, Any]]   # alias / copy of meta_raw_battles

    # Analytics summary (from meta_analytics)
    meta_analytics: Dict[str, Any]

    # Loop / control
    is_balanced: bool                          # True when all conditions are satisfied
    loop_count: int                            # number of "+5 players" loops
    stop_decision: str                         # "enough" | "need_more" | "stop"

    # Logging
    notes: List[str]

    # Standardized participant-level meta table (built after loop)
    meta_table: List[Dict[str, Any]]

    # Compact, LLM-friendly summary tables
    meta_llm_tables: Dict[str, Any]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _plot_meta_matchups_by_deck(
    matchup_summary: List[Dict[str, Any]],
    filename_prefix: str = "meta_matchups",
) -> Dict[str, str]:
    """
    For each deck type (attacker_type), create a bar chart of win rate vs
    every other deck type (defender_type).

    Changes from previous version:
      - Mirror matchups (attacker_type == defender_type) are NOT shown.
      - Bar labels show WR% (not #games).
      - Title includes total games for that deck type.
    """
    if not matchup_summary:
        return {}

    _ensure_plots_dir()

    # Group rows by attacker_type
    by_attacker: Dict[str, List[Dict[str, Any]]] = {}
    for row in matchup_summary:
        attacker = row.get("attacker_type")
        defender = row.get("defender_type")
        if not attacker or not defender:
            continue
        by_attacker.setdefault(attacker, []).append(row)

    plot_paths: Dict[str, str] = {}

    for attacker_type, rows in by_attacker.items():
        if not rows:
            continue

        # Total games for this archetype (including mirror + non-mirror)
        total_games_for_deck = sum(int(r.get("games", 0)) for r in rows)

        # Exclude mirror from what we actually *plot*
        rows_non_mirror = [
            r for r in rows
            if r.get("attacker_type") != r.get("defender_type")
        ]

        if not rows_non_mirror:
            # If there are only mirrors, just skip plotting for this deck
            continue

        # Sort by win_rate descending so strongest matchups appear first
        rows_non_mirror = sorted(
            rows_non_mirror,
            key=lambda r: float(r.get("win_rate", 0.0)),
            reverse=True,
        )

        defenders = [r["defender_type"] for r in rows_non_mirror]
        win_rates_pct = [float(r.get("win_rate", 0.0)) * 100.0 for r in rows_non_mirror]

        x = list(range(len(defenders)))

        plt.figure(figsize=(8, 5))
        plt.bar(x, win_rates_pct)
        plt.xticks(x, defenders, rotation=30, ha="right")
        plt.ylabel("Win rate (%)")
        plt.xlabel("Opponent deck type")

        plt.title(
            f"{attacker_type} vs other deck types "
            f"(meta win rates, {total_games_for_deck} games)"
        )

        # Label each bar with WR%
        for xi, rate in enumerate(win_rates_pct):
            plt.text(
                xi,
                rate + 1.0,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        safe_name = attacker_type.lower().replace(" ", "_")
        filename = f"{filename_prefix}_{safe_name}.png"
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150)
        plt.close()

        plot_paths[attacker_type] = path

    return plot_paths


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def fetch_top_players_node(state: MetaState) -> Dict[str, Any]:
    """
    Fetch top players from the Clash Royale API and initialise state.
    """
    notes = list(state.get("notes", []))

    # Fetch top 1000 player
    top_players = fetch_top_players(limit=1000)
    notes.append(f"Fetched {len(top_players)} top players from API")

    return {
        "top_players": top_players,
        "selected_players": [],
        "used_player_indices": set(),
        "fetched_player_tags": set(),
        "meta_raw_battles": [],
        "normalized_battles": [],
        "meta_analytics": {},
        "meta_table": [],          
        "meta_llm_tables": {},   
        "is_balanced": False,
        "loop_count": 0,
        "stop_decision": "",
        "notes": notes,
    }


def sample_initial_node(state: MetaState) -> Dict[str, Any]:
    """
    Randomly sample 50 players from top_players to form the initial meta cohort.
    """
    top_players = state.get("top_players", [])
    notes = list(state.get("notes", []))

    if not top_players:
        notes.append("sample_initial: WARNING – top_players is empty.")
        return {"selected_players": [], "notes": notes}

    all_indices = list(range(len(top_players)))
    sample_size = min(250, len(all_indices))
    sampled_indices = random.sample(all_indices, sample_size)

    selected_players = [top_players[i] for i in sampled_indices]
    used_indices: Set[int] = set(state.get("used_player_indices", set()))
    used_indices.update(sampled_indices)

    notes.append(
        f"sample_initial: sampled {len(selected_players)} players "
        f"out of {len(top_players)}."
    )

    return {
        "selected_players": selected_players,
        "used_player_indices": used_indices,
        "notes": notes,
    }


def sample_more_5_node(state: MetaState) -> Dict[str, Any]:
    """
    If we still need more battles, sample 5 more *unused* players from top_players.
    """
    top_players = state.get("top_players", [])
    used_indices: Set[int] = set(state.get("used_player_indices", set()))
    loop_count = state.get("loop_count", 0)
    notes = list(state.get("notes", []))

    if not top_players:
        notes.append("sample_more_5: WARNING – top_players is empty.")
        return {"selected_players": [], "notes": notes}

    all_indices = list(range(len(top_players)))
    unused_indices = [i for i in all_indices if i not in used_indices]

    if not unused_indices:
        notes.append("sample_more_5: no unused players left; cannot sample more.")
        return {
            "selected_players": [],
            "used_player_indices": used_indices,
            "notes": notes,
        }

    sample_size = min(5, len(unused_indices))
    new_indices = random.sample(unused_indices, sample_size)
    selected_players = [top_players[i] for i in new_indices]

    used_indices.update(new_indices)
    loop_count += 1

    notes.append(
        f"sample_more_5: loop {loop_count} – sampled {len(selected_players)} more "
        f"players; total_used={len(used_indices)}/{len(top_players)}."
    )

    return {
        "selected_players": selected_players,
        "used_player_indices": used_indices,
        "loop_count": loop_count,
        "notes": notes,
    }


def fetch_meta_battles_node(state: MetaState) -> Dict[str, Any]:
    """
    For each selected player, fetch their battlelog and add up to the 10 most
    recent ranked 1v1 battles (normalized) to meta_raw_battles.
    """
    selected = state.get("selected_players", [])
    notes = list(state.get("notes", []))
    meta_raw = list(state.get("meta_raw_battles", []))
    fetched_tags: Set[str] = set(state.get("fetched_player_tags", set()))

    if not selected:
        notes.append("fetch_meta_battles: no selected_players; nothing to fetch.")
        return {
            "meta_raw_battles": meta_raw,
            "normalized_battles": meta_raw,
            "fetched_player_tags": fetched_tags,
            "notes": notes,
        }

    new_battle_count = 0
    new_player_count = 0

    for player in selected:
        tag = player.get("tag")
        if not tag:
            continue

        if tag in fetched_tags:
            # Already fetched this player in a previous loop
            continue

        try:
            raw_log = get_player_battlelog(tag)
            normalized = filter_and_normalize_ranked_1v1(raw_log)

            # Take up to 10 most recent ranked 1v1 games
            take_n = min(len(normalized), 10)
            recent_ranked = normalized[:take_n]

            meta_raw.extend(recent_ranked)
            new_battle_count += len(recent_ranked)
            new_player_count += 1
            fetched_tags.add(tag)

        except Exception as e:
            notes.append(
                f"fetch_meta_battles: error fetching {tag}: {str(e)}"
            )

    notes.append(
        "fetch_meta_battles: fetched "
        f"{new_battle_count} normalized ranked 1v1 battles "
        f"from {new_player_count} new players. "
        f"total_meta_battles={len(meta_raw)}"
    )

    # normalized_battles mirrors meta_raw_battles for now
    return {
        "meta_raw_battles": meta_raw,
        "normalized_battles": meta_raw,
        "fetched_player_tags": fetched_tags,
        "notes": notes,
    }


def compute_meta_analytics_node(state: MetaState) -> Dict[str, Any]:
    """
    Run the meta analytics engine on all normalized battles.
    """
    battles = state.get("meta_raw_battles", [])
    analytics = compute_meta_analytics(battles)

    notes = list(state.get("notes", []))
    notes.append(
        f"compute_meta_analytics: games_total={analytics.get('games_total', len(battles))}, "
        f"deck_types_opp={len(analytics.get('opp_deck_types', []))}"
    )

    return {
        "meta_analytics": analytics,
        "notes": notes,
    }


def check_enough_battles_node(state: MetaState) -> Dict[str, Any]:
    """
    Stopping condition for Phase 0:

    1. Total games >= MIN_TOTAL_BATTLES (500)
    2. For each required deck type (siege, bait, cycle, bridge spam, beatdown),
       the **combined** sample size (my + opp decks) >= MIN_GAMES_PER_TYPE.

    Hybrid is allowed to have fewer than MIN_GAMES_PER_TYPE games.

    This node sets:
        - is_balanced: bool
        - stop_decision: "enough" | "need_more" | "stop"
    """
    notes = list(state.get("notes", []))
    meta = state.get("meta_analytics", {}) or {}

    # Use summary.games_played if present, otherwise fall back to raw battle length
    summary = meta.get("summary", {}) or {}
    games_total = int(
        summary.get("games_played", len(state.get("meta_raw_battles", [])))
    )

    # --- NEW: combine my + opp deck-type counts ---
    my_counts_raw = meta.get("deck_type_counts_my", {}) or {}
    opp_counts_raw = meta.get("deck_type_counts_opp", {}) or {}

    combined_counts_raw: Dict[str, int] = {}
    all_keys = set(my_counts_raw.keys()) | set(opp_counts_raw.keys())
    for k in all_keys:
        combined_counts_raw[k] = int(my_counts_raw.get(k, 0)) + int(
            opp_counts_raw.get(k, 0)
        )

    # Normalize deck-type keys to lowercase for robustness
    deck_counts_lower: Dict[str, int] = {
        str(k).lower(): int(v) for k, v in combined_counts_raw.items()
    }

    # Check required deck types
    insufficient_types: Dict[str, int] = {}
    for t in REQUIRED_DECK_TYPES_LOWER:
        count = deck_counts_lower.get(t, 0)
        if count < MIN_GAMES_PER_TYPE:
            insufficient_types[t] = count

    enough_total = games_total >= MIN_TOTAL_BATTLES
    enough_per_type = len(insufficient_types) == 0

    # Decide what to do next
    top_players = state.get("top_players", [])
    used_indices: Set[int] = set(state.get("used_player_indices", set()))
    remaining = max(0, len(top_players) - len(used_indices))
    loop_count = state.get("loop_count", 0)

    if enough_total and enough_per_type:
        decision = "enough"
        notes.append(
            "check_enough_battles: enough data. "
            f"games_total={games_total}, all required deck types >= {MIN_GAMES_PER_TYPE} "
            "(combined my+opp)."
        )
        is_balanced = True
    else:
        # If we can't or shouldn't loop more, stop with what we have
        if remaining <= 0 or loop_count >= 20:
            decision = "stop"
            notes.append(
                "check_enough_battles: stopping. "
                f"games_total={games_total}, remaining_players={remaining}, "
                f"loop_count={loop_count}, insufficient_types={insufficient_types}."
            )
            is_balanced = False
        else:
            decision = "need_more"
            notes.append(
                "check_enough_battles: need more data. "
                f"games_total={games_total}, remaining_players={remaining}, "
                f"loop_count={loop_count}, insufficient_types={insufficient_types}."
            )
            is_balanced = False

    return {
        "is_balanced": is_balanced,
        "stop_decision": decision,
        "notes": notes,
    }

def standardize_meta_table_node(state: MetaState) -> Dict[str, Any]:
    """
    After we've finished looping, build a unified participant-level meta table
    from all normalized meta battles.

    This does NOT run inside the loop; it only runs once when we decide the
    dataset is "enough" or when we stop due to loop/players limits.
    """
    notes = list(state.get("notes", []))
    battles = state.get("meta_raw_battles", []) or []

    if not battles:
        notes.append("standardize_meta_table: no battles in state, skipping.")
        return {"notes": notes}

    meta_table = build_standardized_meta_table(battles)
    notes.append(
        f"standardize_meta_table: built meta_table with {len(meta_table)} rows "
        f"from {len(battles)} battles."
    )

    return {
        "meta_table": meta_table,
        "notes": notes,
    }

def _aggregate_meta_deck_type_stats(
    meta_table: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Aggregate participant-level meta table into deck-type stats compatible with
    plot_deck_type_pie / plot_deck_type_bar.

    Output list entries look like:
        {
          "type": "Beatdown",
          "games": 42,
          "wins": 23,
          "losses": 17,
          "draws": 2,
          "win_rate": 23/42,
        }
    """
    stats: Dict[str, Dict[str, Any]] = {}

    for row in meta_table:
        deck_type = row.get("deck_type", "Unknown")
        result = row.get("result")

        rec = stats.setdefault(
            deck_type,
            {
                "type": deck_type,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.0,
            },
        )

        rec["games"] += 1
        if result == "win":
            rec["wins"] += 1
        elif result == "loss":
            rec["losses"] += 1
        elif result == "draw":
            rec["draws"] += 1

    for rec in stats.values():
        games = rec["games"] or 0
        rec["win_rate"] = rec["wins"] / games if games > 0 else 0.0

    # Sort by games descending so plots are stable and readable
    return sorted(stats.values(), key=lambda r: r["games"], reverse=True)

def build_meta_llm_tables_node(state: MetaState) -> Dict[str, Any]:
    """
    Build compact, LLM-friendly meta tables from the finalized meta dataset.

    Inputs:
      - state["meta_table"]: participant-level rows built after the loop
      - state["meta_analytics"]["deck_type_matchups"]: full matchup matrix

    Outputs:
      - state["meta_llm_tables"] = {
            "meta_deck_summary": [...],
            "meta_matchup_summary": [...],
        }
    """
    notes = list(state.get("notes", []))
    meta_table = state.get("meta_table", []) or []
    analytics = state.get("meta_analytics", {}) or {}
    matchups = analytics.get("deck_type_matchups", {}) or {}

    if not meta_table:
        notes.append(
            "build_meta_llm_tables: no meta_table available, skipping LLM tables."
        )
        return {"notes": notes}

    # Use the same MIN_GAMES_PER_TYPE as your loop for sample_ok
    deck_summary = build_meta_deck_summary(
        meta_table,
        min_games_per_type=MIN_GAMES_PER_TYPE,
    )
    matchup_summary = build_meta_matchup_summary(
        matchups,
        min_matchup_games=30,  # you can tune this if you want stricter matchup samples
    )

    llm_tables: Dict[str, Any] = {
        "meta_deck_summary": deck_summary,
        "meta_matchup_summary": matchup_summary,
    }

    notes.append(
        "build_meta_llm_tables: built "
        f"{len(deck_summary)} deck-type rows and "
        f"{len(matchup_summary)} matchup rows."
    )

    return {
        "meta_llm_tables": llm_tables,
        "notes": notes,
    }


def generate_meta_plots_node(state: MetaState) -> Dict[str, Any]:
    """
    Generate ALL meta plots AFTER the loop has finished.

    Uses:
      - state["meta_llm_tables"]["meta_deck_summary"]
      - state["meta_llm_tables"]["meta_matchup_summary"]

    Writes:
      - meta_analytics["plots"] = {
            "meta_deck_types_pie": ...,
            "meta_deck_types_winrate_bar": ...,
            "meta_matchups_by_deck": {
                "<deck_type>": "<path>",
                ...
            },
        }
    """
    notes = list(state.get("notes", []))

    analytics = dict(state.get("meta_analytics", {}) or {})
    plots = dict(analytics.get("plots", {}) or {})

    llm_tables = state.get("meta_llm_tables", {}) or {}
    deck_summary = llm_tables.get("meta_deck_summary", []) or []
    matchup_summary = llm_tables.get("meta_matchup_summary", []) or []

    # --- 1) Overall meta deck-type plots (pie + win-rate bar) ---

    if deck_summary:
        # plots.plot_deck_type_* expects "type" instead of "deck_type"
        deck_types_for_plots: List[Dict[str, Any]] = [
            {
                "type": row.get("deck_type", "Unknown"),
                "games": int(row.get("games", 0)),
                "wins": int(row.get("wins", 0)),
                "losses": int(row.get("losses", 0)),
                "draws": int(row.get("draws", 0)),
                "win_rate": float(row.get("win_rate", 0.0)),
            }
            for row in deck_summary
        ]

        plots["meta_deck_types_pie"] = plot_deck_type_pie(
            deck_types_for_plots,
            title="Meta Deck Types (by Games Played)",
            filename="meta_deck_types",
        )

        plots["meta_deck_types_winrate_bar"] = plot_deck_type_bar(
            deck_types_for_plots,
            title="Meta Deck Types Win Rate (All Participants)",
            filename="meta_deck_types_winrate",
            metric="win_rate",
        )

        notes.append(
            f"generate_meta_plots: created meta_deck_types_pie and "
            f"meta_deck_types_winrate_bar for {len(deck_types_for_plots)} deck types."
        )
    else:
        notes.append(
            "generate_meta_plots: no meta_deck_summary in meta_llm_tables; "
            "skipping deck-type pie/bar plots."
        )

    # --- 2) Per-deck matchup graphs: each deck vs other types (W/R) ---

    if matchup_summary:
        per_deck_paths = _plot_meta_matchups_by_deck(
            matchup_summary,
            filename_prefix="meta_matchups",
        )
        plots["meta_matchups_by_deck"] = per_deck_paths

        notes.append(
            "generate_meta_plots: generated per-deck matchup charts for "
            f"{len(per_deck_paths)} deck types."
        )
    else:
        notes.append(
            "generate_meta_plots: no meta_matchup_summary in meta_llm_tables; "
            "skipping per-deck matchup plots."
        )

    # Save plots back into analytics
    analytics["plots"] = plots

    return {
        "meta_analytics": analytics,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def route_after_check_enough(state: MetaState) -> str:
    """
    Decide which edge to take after check_enough_battles_node.

    Returns one of:
      - "enough"    -> we have enough games and deck-type coverage; finish.
      - "need_more" -> we need more games; sample more players.
      - "stop"      -> cannot continue (no players / too many loops); stop.
    """
    decision = state.get("stop_decision", "")
    if decision in ("enough", "need_more", "stop"):
        return decision

    # Fallback: if something goes weird, just stop
    return "stop"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_meta_graph():
    """
    Build the Phase 0 LangGraph for meta dataset construction.

    Flow:

        fetch_top_players
            ↓
        sample_initial
            ↓
        fetch_meta_battles
            ↓
        compute_meta_analytics
            ↓
        check_enough_battles ──(enough/stop)──▶ END
                     │
                     └── need_more ──▶ sample_more_5 ──▶ fetch_meta_battles (loop)
    """
    graph = StateGraph(MetaState)

    #intial node
    graph.add_node("fetch_top_players", fetch_top_players_node)

    graph.add_node("sample_initial", sample_initial_node)
    graph.add_node("fetch_meta_battles", fetch_meta_battles_node)
    graph.add_node("compute_meta_analytics", compute_meta_analytics_node)
    graph.add_node("check_enough_battles", check_enough_battles_node)
    graph.add_node("sample_more_5", sample_more_5_node)

    # post-loop nodes:
    graph.add_node("standardize_meta_table", standardize_meta_table_node)
    graph.add_node("build_meta_llm_tables", build_meta_llm_tables_node)
    graph.add_node("generate_meta_plots", generate_meta_plots_node)

    #intal starting
    graph.set_entry_point("fetch_top_players")

    # Loop
    graph.add_edge("fetch_top_players", "sample_initial")
    graph.add_edge("sample_initial", "fetch_meta_battles")
    graph.add_edge("sample_more_5", "fetch_meta_battles")
    graph.add_edge("fetch_meta_battles", "compute_meta_analytics")
    graph.add_edge("compute_meta_analytics", "check_enough_battles")

    # After we stop looping:
    graph.add_edge("standardize_meta_table", "build_meta_llm_tables")
    graph.add_edge("build_meta_llm_tables", "generate_meta_plots")
    graph.add_edge("generate_meta_plots", END)

    graph.add_conditional_edges(
        "check_enough_battles",
        route_after_check_enough,
        {
            "enough": "standardize_meta_table",
            "stop": "standardize_meta_table",
            "need_more": "sample_more_5",
        },
    )

    return graph.compile()
