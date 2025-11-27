# src/analytics/meta_analytics.py

from typing import Any, Dict, List

from .user_analytics import compute_user_analytics


def compute_meta_analytics(
    battles_normalized: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute analytics for the *meta* dataset.

    This is a thin wrapper around compute_user_analytics so that
    meta_analytics and user_analytics share the same structure:

        {
          "summary": {...},
          "best_cards": [...],
          "worst_cards": [...],
          "tough_opp_cards": [...],
          "easy_opp_cards": [...],
          "best_decks": [...],
          "worst_decks": [...],
          "tough_matchups": [...],
          "easy_matchups": [...],
          "my_deck_types": [...],
          "opp_deck_types": [...],
          "plots": {...},
          ...
        }

    We add:
        - games_total
        - deck_type_counts_opp: { deck_type: games }
        - source: "meta"
    """
    analytics = compute_user_analytics(battles_normalized)

    # Total games: prefer summary if present
    summary = analytics.get("summary", {})
    games_total = int(summary.get("games_played", len(battles_normalized)))
    analytics["games_total"] = games_total

    # Opponent deck-type game counts for Phase 0 stopping condition
    opp_deck_types = analytics.get("opp_deck_types", [])
    deck_type_counts_opp: Dict[str, int] = {
        str(row.get("type", "")).strip(): int(row.get("games", 0))
        for row in opp_deck_types
    }
    analytics["deck_type_counts_opp"] = deck_type_counts_opp

    # Tag this analytics object as coming from the meta dataset
    analytics["source"] = "meta"

    return analytics
