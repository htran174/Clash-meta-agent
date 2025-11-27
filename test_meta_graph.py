# test_meta_graph.py
"""
Quick driver to run Phase 0 (meta graph) and print a summary.

Usage:
    python test_meta_graph.py
"""

from typing import Dict, Any

from src.workflows.meta_workflow import build_meta_graph, MetaState


def main() -> None:
    # Build the Phase 0 graph
    graph = build_meta_graph()

    # Initial state: graph fills in everything else
    initial_state: MetaState = {
        "notes": [],
    }

    # Allow enough loop iterations for the "+5 players" logic
    config: Dict[str, Any] = {
        "recursion_limit": 80,
    }

    final_state: MetaState = graph.invoke(initial_state, config=config)

    meta_raw = final_state.get("meta_raw_battles", [])
    meta_analytics: Dict[str, Any] = final_state.get("meta_analytics", {}) or {}

    # Try a few obvious keys for deck-type counts, depending on how we stored them
    deck_counts_opp = (
        meta_analytics.get("deck_type_counts_opp")
        or meta_analytics.get("deck_type_counts")
        or {}
    )
    deck_counts_my = meta_analytics.get("deck_type_counts_my", {})

    games_total = meta_analytics.get("games_total", len(meta_raw))

    print("Top players:", len(final_state.get("top_players", [])))
    print("Selected players:", len(final_state.get("selected_players", [])))
    print("Meta raw battles:", len(meta_raw))
    print("Games total (from analytics):", games_total)
    print("Loop count:", final_state.get("loop_count", 0))
    print("Stop decision:", final_state.get("stop_decision", ""))
    print("Is balanced:", final_state.get("is_balanced", False))

    # --- Deck-type counts ---
    print("\n=== Deck-type counts (opponent decks) ===")
    if deck_counts_opp:
        for deck_type, count in sorted(deck_counts_opp.items(), key=lambda x: x[0].lower()):
            print(f"  {deck_type:12s} -> {count:4d}")
    else:
        print("  (no deck_type_counts_opp found in meta_analytics)")

    if deck_counts_my:
        print("\n=== Deck-type counts (my decks) ===")
        for deck_type, count in sorted(deck_counts_my.items(), key=lambda x: x[0].lower()):
            print(f"  {deck_type:12s} -> {count:4d}")

    # --- Notes / debug trail ---
    print("\n=== Notes ===")
    for note in final_state.get("notes", []):
        print(" -", note)


if __name__ == "__main__":
    main()
