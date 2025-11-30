# src/analytics/meta_standardize.py

from __future__ import annotations

from typing import Any, Dict, List

from . import deck_type as deck_type_mod


def _classify_deck(cards: List[str]) -> str:
    """
    Same wrapper pattern as in meta_analytics.py so don't depend
    on the exact function name in deck_type.py.
    """
    if hasattr(deck_type_mod, "classify_deck_from_cards"):
        return deck_type_mod.classify_deck_from_cards(cards)  # type: ignore[attr-defined]
    if hasattr(deck_type_mod, "classify_deck"):
        return deck_type_mod.classify_deck(cards)  # type: ignore[attr-defined]
    return "Hybrid"


def _flip_result(res: str) -> str:
    """Flip a result when swap POV: win<->loss, draw stays draw."""
    if res == "win":
        return "loss"
    if res == "loss":
        return "win"
    return "draw"


def build_standardized_meta_table(
    normalized_battles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build a unified meta table with one row per participant.

    Input: same normalized battles pass into compute_meta_analytics, with at least:
      - 'result'    : 'win' | 'loss' | 'draw' (from *my* POV)
      - 'my_cards'  : List[str] (top player's deck)
      - 'opp_cards' : List[str] (opponent's deck)

    Output: list of rows like:
      {
        "battle_index": int,
        "deck_type": "Cycle" | "Beatdown" | ...,
        "role": "my" | "opp",
        "result": "win" | "loss" | "draw",
        "is_win": bool,
      }

    This is the canonical table use for meta plots (no “my vs opp” labels,
    just “participants in the meta”).
    """
    rows: List[Dict[str, Any]] = []

    for idx, battle in enumerate(normalized_battles):
        result = battle.get("result")
        my_cards = battle.get("my_cards")
        opp_cards = battle.get("opp_cards")

        # Skip malformed entries
        if not isinstance(my_cards, list) or not isinstance(opp_cards, list):
            continue
        if result not in ("win", "loss", "draw"):
            continue

        # Try to keep a stable id if one exists, otherwise use index
        battle_id = battle.get("battle_id", idx)

        # Classify both decks type
        my_type = _classify_deck(my_cards)
        opp_type = _classify_deck(opp_cards)

        # Row for the "my" side (top player)
        rows.append(
            {
                "battle_id": battle_id,
                "battle_index": idx,
                "deck_type": my_type,
                "role": "my",
                "result": result,
                "is_win": result == "win",
            }
        )

        # Row for the opponent side (flip POV)
        opp_result = _flip_result(result)
        rows.append(
            {
                "battle_id": battle_id,
                "battle_index": idx,
                "deck_type": opp_type,
                "role": "opp",
                "result": opp_result,
                "is_win": opp_result == "win",
            }
        )

    return rows
