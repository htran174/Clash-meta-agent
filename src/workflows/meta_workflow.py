# src/workflows/meta_workflow.py

from __future__ import annotations

import random
from typing import Any, Dict, List, Set
from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.api.players import fetch_top_players
from src.api.battles import get_player_battlelog
from src.analytics.battle_filters import filter_and_normalize_ranked_1v1
from src.analytics.meta_analytics import compute_meta_analytics


# ---------------------------------------------------------------------------
# Constants for Phase 0 stopping condition
# ---------------------------------------------------------------------------

MIN_TOTAL_BATTLES = 500
MIN_GAMES_PER_TYPE = 50

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


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def fetch_top_players_node(state: MetaState) -> Dict[str, Any]:
    """
    Fetch top players from the Clash Royale API and initialise state.
    """
    notes = list(state.get("notes", []))

    # You modified this function to accept an optional limit
    top_players = fetch_top_players(limit=300)
    notes.append(f"Fetched {len(top_players)} top players from API")

    return {
        "top_players": top_players,
        "selected_players": [],
        "used_player_indices": set(),
        "fetched_player_tags": set(),
        "meta_raw_battles": [],
        "normalized_battles": [],
        "meta_analytics": {},
        "is_balanced": False,
        "loop_count": 0,
        "stop_decision": "",
        "notes": notes,
    }


def sample_initial_50_node(state: MetaState) -> Dict[str, Any]:
    """
    Randomly sample 50 players from top_players to form the initial meta cohort.
    """
    top_players = state.get("top_players", [])
    notes = list(state.get("notes", []))

    if not top_players:
        notes.append("sample_initial_50: WARNING – top_players is empty.")
        return {"selected_players": [], "notes": notes}

    all_indices = list(range(len(top_players)))
    sample_size = min(50, len(all_indices))
    sampled_indices = random.sample(all_indices, sample_size)

    selected_players = [top_players[i] for i in sampled_indices]
    used_indices: Set[int] = set(state.get("used_player_indices", set()))
    used_indices.update(sampled_indices)

    notes.append(
        f"sample_initial_50: sampled {len(selected_players)} players "
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
       opponent deck-type sample size >= MIN_GAMES_PER_TYPE (70).

    Hybrid is allowed to have fewer than 70 games.

    This node sets:
        - is_balanced: bool
        - stop_decision: "enough" | "need_more" | "stop"
    """
    notes = list(state.get("notes", []))
    meta = state.get("meta_analytics", {}) or {}

    games_total = int(meta.get("games_total", len(state.get("meta_raw_battles", []))))
    deck_counts_raw = meta.get("deck_type_counts_opp", {})

    # Normalize deck-type keys to lowercase for robustness
    deck_counts_lower: Dict[str, int] = {
        str(k).lower(): int(v) for k, v in deck_counts_raw.items()
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
            f"games_total={games_total}, all required deck types >= {MIN_GAMES_PER_TYPE}."
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
        sample_initial_50
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

    # Nodes
    graph.add_node("fetch_top_players", fetch_top_players_node)
    graph.add_node("sample_initial_50", sample_initial_50_node)
    graph.add_node("fetch_meta_battles", fetch_meta_battles_node)
    graph.add_node("compute_meta_analytics", compute_meta_analytics_node)
    graph.add_node("check_enough_battles", check_enough_battles_node)
    graph.add_node("sample_more_5", sample_more_5_node)

    # Entry
    graph.set_entry_point("fetch_top_players")

    # Linear edges
    graph.add_edge("fetch_top_players", "sample_initial_50")
    graph.add_edge("sample_initial_50", "fetch_meta_battles")
    graph.add_edge("sample_more_5", "fetch_meta_battles")
    graph.add_edge("fetch_meta_battles", "compute_meta_analytics")
    graph.add_edge("compute_meta_analytics", "check_enough_battles")

    # Conditional routing after checking battle count + deck types
    graph.add_conditional_edges(
        "check_enough_battles",
        route_after_check_enough,
        {
            "enough": END,           # we have >= 500 battles AND each required type >= 70
            "stop": END,             # no more players / too many loops
            "need_more": "sample_more_5",
        },
    )

    return graph.compile()
