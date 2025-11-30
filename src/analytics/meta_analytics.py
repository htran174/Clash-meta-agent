#src/analytics/meta_analytics.py
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from . import deck_type as deck_type_mod

# Sync with archetypes in deck_type.py
DECK_TYPES: List[str] = ["Bait", "Beatdown", "Bridge Spam", "Cycle", "Hybrid", "Siege"]


def _classify_deck(cards: List[str]) -> str:
    """
    Small wrapper so we don't depend on an exact function name in deck_type.py.

    Prefers:
        - classify_deck_from_cards(cards)
    Fallback:
        - classify_deck(cards)
    Final fallback:
        - "Hybrid"
    """
    if hasattr(deck_type_mod, "classify_deck_from_cards"):
        return deck_type_mod.classify_deck_from_cards(cards)  # type: ignore[attr-defined]
    if hasattr(deck_type_mod, "classify_deck"):
        return deck_type_mod.classify_deck(cards)  # type: ignore[attr-defined]
    return "Hybrid"


def _flip_result(res: str) -> str:
    """Flip a result when we swap POV: win<->loss, draw stays draw."""
    if res == "win":
        return "loss"
    if res == "loss":
        return "win"
    return "draw"


def _build_symmetric_matchup_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build a deck-type vs deck-type matchup table that counts BOTH sides.

    For each game we create two rows:
      1) my_deck_type  vs opp_deck_type, result = result
      2) opp_deck_type vs my_deck_type, result = flipped(result)

    So 'Bait' stats include all games where Bait showed up, regardless of side.
    """
    if df.empty:
        return {}

    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        my_type = row["my_deck_type"]
        opp_type = row["opp_deck_type"]
        res = row["result"]

        # POV: my deck (ranker)
        rows.append(
            {"deck_type": my_type, "opp_type": opp_type, "result": res}
        )

        # POV: opponent deck
        rows.append(
            {
                "deck_type": opp_type,
                "opp_type": my_type,
                "result": _flip_result(res),
            }
        )

    tmp = pd.DataFrame(rows)

    group = tmp.groupby(["deck_type", "opp_type"])["result"]
    agg = group.agg(
        games="count",
        wins=lambda s: (s == "win").sum(),
        losses=lambda s: (s == "loss").sum(),
        draws=lambda s: (s == "draw").sum(),
    ).reset_index()

    # Avoid division by zero
    agg["win_rate"] = agg["wins"] / agg["games"].where(agg["games"] > 0, 1)

    matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for _, row in agg.iterrows():
        deck_type: str = row["deck_type"]
        opp_type: str = row["opp_type"]

        matrix.setdefault(deck_type, {})[opp_type] = {
            "games": int(row["games"]),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "draws": int(row["draws"]),
            "win_rate": float(row["win_rate"]),
        }

    return matrix


def compute_meta_analytics(normalized_battles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute Phase 0 meta analytics from a list of normalized battles.

    Each battle dict is expected to at least have:
      - 'result'    : 'win' | 'loss' | 'draw' (from *my* POV)
      - 'my_cards'  : List[str] of 8 card names (top player's deck)
      - 'opp_cards' : List[str] of 8 card names (opponent's deck)

    Returns a dict with (at minimum):

    analytics = {
        "summary": {
            "games_played": int,
            "wins": int,
            "losses": int,
            "draws": int,
            "win_rate": float,
        },
        "deck_type_counts_my": { archetype: count, ... },
        "deck_type_counts_opp": { archetype: count, ... },
        "deck_type_matchups": {
            deck_type: {
                opp_type: {
                    "games": int,
                    "wins": int,
                    "losses": int,
                    "draws": int,
                    "win_rate": float,
                },
                ...
            },
            ...
        },
    }

    Phase 0 graph currently reads:
      - summary["games_played"]
      - deck_type_counts_opp
    Those keys are preserved; only deck_type_matchups logic is new.
    """
    if not normalized_battles:
        empty_summary = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "win_rate": 0.0,
        }
        return {
            "summary": empty_summary,
            "deck_type_counts_my": {},
            "deck_type_counts_opp": {},
            "deck_type_matchups": {},
        }

    df = pd.DataFrame(normalized_battles)

    # --- Basic summary ---
    if "result" not in df.columns:
        raise ValueError("normalized_battles must include a 'result' field")

    wins = int((df["result"] == "win").sum())
    losses = int((df["result"] == "loss").sum())
    draws = int((df["result"] == "draw").sum())
    games = int(len(df))

    win_rate = float(wins / games) if games > 0 else 0.0

    summary = {
        "games_played": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
    }

    # --- Deck-type classification for BOTH sides ---
    if "my_cards" not in df.columns or "opp_cards" not in df.columns:
        raise ValueError("normalized_battles must include 'my_cards' and 'opp_cards'")

    df = df.copy()  # don't mutate caller's DataFrame
    df["my_deck_type"] = df["my_cards"].apply(_classify_deck)
    df["opp_deck_type"] = df["opp_cards"].apply(_classify_deck)

    # --- Count appearances separately for my / opp (useful for sanity checks) ---
    my_counts = df["my_deck_type"].value_counts().to_dict()
    opp_counts = df["opp_deck_type"].value_counts().to_dict()

    # Ensure all known deck types are present (with 0) for easier downstream logic
    for archetype in DECK_TYPES:
        my_counts.setdefault(archetype, 0)
        opp_counts.setdefault(archetype, 0)

    # --- Symmetric matchup matrix: deck_type vs opp_type, using BOTH sides ---
    deck_type_matchups = _build_symmetric_matchup_matrix(df)

    analytics: Dict[str, Any] = {
        "summary": summary,
        "deck_type_counts_my": my_counts,
        "deck_type_counts_opp": opp_counts,
        "deck_type_matchups": deck_type_matchups,
    }

    return analytics
