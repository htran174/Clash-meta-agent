#src/analytics/user_analytics.py
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd

from .deck_type import summarize_deck_types, classify_deck

def compute_deck_type_matchups(
    battles_normalized: List[Dict[str, Any]],
    min_games: int = 1,
) -> List[Dict[str, Any]]:
    """
    Compute user performance by (my_deck_type, opp_deck_type).

    Each battle is classified into:
      - my_deck_type  = archetype of my_cards
      - opp_deck_type = archetype of opp_cards

    We aggregate counts for each (my_deck_type, opp_deck_type) pair.

    Returns a list of rows like:

        {
          "my_deck_type": "Beatdown",
          "opp_deck_type": "Bridge Spam",
          "games": 6,
          "wins": 1,
          "losses": 5,
          "draws": 0,
          "win_rate": 1/6,
        }
    """
    stats: Dict[Tuple[str, str], Dict[str, int]] = {}

    def _ensure_bucket(key: Tuple[str, str]) -> Dict[str, int]:
        if key not in stats:
            stats[key] = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        return stats[key]

    for battle in battles_normalized:
        result = battle.get("result")
        my_cards = battle.get("my_cards") or []
        opp_cards = battle.get("opp_cards") or []

        # Only trust full 8-card decks
        try:
            my_type = classify_deck(my_cards) if len(my_cards) == 8 else None
        except Exception:
            my_type = None

        try:
            opp_type = classify_deck(opp_cards) if len(opp_cards) == 8 else None
        except Exception:
            opp_type = None

        if my_type is None or opp_type is None:
            continue

        key = (my_type, opp_type)
        bucket = _ensure_bucket(key)
        bucket["games"] += 1
        if result == "win":
            bucket["wins"] += 1
        elif result == "loss":
            bucket["losses"] += 1
        else:
            bucket["draws"] += 1

    # Convert to list of dicts with win_rate
    out: List[Dict[str, Any]] = []
    for (my_type, opp_type), s in stats.items():
        if s["games"] < min_games:
            continue

        games = s["games"]
        wins = s["wins"]
        losses = s["losses"]
        draws = s["draws"]
        win_rate = wins / games if games > 0 else 0.0

        out.append(
            {
                "my_deck_type": my_type,
                "opp_deck_type": opp_type,
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate,
            }
        )

    # Sort: more games first, then higher winrate
    out.sort(key=lambda d: (d["games"], d["win_rate"]), reverse=True)
    return out


def compute_user_deck_matchups(
    battles_normalized: List[Dict[str, Any]],
    overall_win_rate: float,
    min_games: int = 1,
    winrate_delta: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute tough/easy matchups from the *user's* perspective by grouping
    battles by opponent deck (exact 8-card list).

    Returns:
        (tough_matchups, easy_matchups)

    Each row looks like:
        {
          "deck": [... 8 card names ...],
          "games": int,
          "wins": int,   # your wins vs this deck
          "losses": int, # your losses vs this deck
          "draws": int,
          "win_rate": float,  # your winrate vs this deck
        }
    """
    # Aggregate stats per opponent deck, from *your* perspective
    opp_decks: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    )

    for b in battles_normalized:
        result = b.get("result")
        opp_cards = b.get("opp_cards") or []
        if not opp_cards:
            continue

        deck_key = tuple(sorted(opp_cards))
        s = opp_decks[deck_key]
        s["games"] += 1
        if result == "win":
            s["wins"] += 1          # you won vs this deck
        elif result == "loss":
            s["losses"] += 1        # you lost vs this deck
        else:
            s["draws"] += 1

    # Convert to list with win_rate, filter by min_games
    rows: List[Dict[str, Any]] = []
    for deck_key, s in opp_decks.items():
        if s["games"] < min_games:
            continue
        wr = s["wins"] / s["games"] if s["games"] > 0 else 0.0
        rows.append(
            {
                "deck": list(deck_key),
                "games": s["games"],
                "wins": s["wins"],
                "losses": s["losses"],
                "draws": s["draws"],
                "win_rate": wr,
            }
        )

    # Split into tough vs easy based on your overall winrate
    tough: List[Dict[str, Any]] = []
    easy: List[Dict[str, Any]] = []

    for row in rows:
        wr = row["win_rate"]
        if wr <= overall_win_rate - winrate_delta:
            tough.append(row)
        if wr >= overall_win_rate + winrate_delta:
            easy.append(row)

    # Sort each list by how extreme the matchup is
    tough.sort(key=lambda r: (r["win_rate"], r["games"]))          # worst first
    easy.sort(key=lambda r: (r["win_rate"], r["games"]), reverse=True)  # best first

    return tough, easy


def build_battles_dataframe(
    battles_normalized: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Turn a list of normalized battle dicts into a pandas DataFrame.

    Expected battle schema:
        {
          "battle_time": str,
          "result": "win" | "loss" | "draw",
          "my_cards": List[str],
          "opp_cards": List[str],
          "mode_name": str,
        }
    """
    if not battles_normalized:
        return pd.DataFrame(
            columns=["battle_time", "result", "my_cards", "opp_cards", "mode_name"]
        )

    df = pd.DataFrame(battles_normalized)

    for col in ["battle_time", "result", "my_cards", "opp_cards", "mode_name"]:
        if col not in df.columns:
            df[col] = None

    return df


# ---------- Overall summary ----------


def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall performance summary."""
    total_games = len(df)
    if total_games == 0:
        return {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "win_rate": 0.0,
        }

    wins = int((df["result"] == "win").sum())
    losses = int((df["result"] == "loss").sum())
    draws = int((df["result"] == "draw").sum())

    win_rate = wins / total_games if total_games > 0 else 0.0

    return {
        "games_played": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
    }


# ---------- Card-level stats ----------


def _card_stats_from_rows(
    rows: List[Dict[str, Any]],
    min_games: int = 3,
    sort_desc: bool = True,
) -> List[Dict[str, Any]]:
    """
    Helper to build card stats from rows like:
        {"card": "Card Name", "result": "win" | "loss" | "draw"}
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    )

    for row in rows:
        card = row["card"]
        result = row["result"]
        s = stats[card]
        s["games"] += 1
        if result == "win":
            s["wins"] += 1
        elif result == "loss":
            s["losses"] += 1
        else:
            s["draws"] += 1

    out: List[Dict[str, Any]] = []
    for card, s in stats.items():
        if s["games"] < min_games:
            continue
        wr = s["wins"] / s["games"] if s["games"] > 0 else 0.0
        out.append(
            {
                "card": card,
                "games": s["games"],
                "wins": s["wins"],
                "losses": s["losses"],
                "draws": s["draws"],
                "win_rate": wr,
            }
        )

    out.sort(key=lambda x: (x["win_rate"], x["games"]), reverse=sort_desc)
    return out


def compute_card_performance(df: pd.DataFrame, min_games: int = 3) -> Dict[str, Any]:
    """
    Compute card-level performance for:
      - my cards  (best/worst)
      - opponent cards (tough/easy)
    """
    rows_my: List[Dict[str, Any]] = []
    rows_opp: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        result = row["result"]

        for card in row.get("my_cards", []) or []:
            rows_my.append({"card": card, "result": result})

        for card in row.get("opp_cards", []) or []:
            if result == "win":
                opp_result = "loss"
            elif result == "loss":
                opp_result = "win"
            else:
                opp_result = "draw"
            rows_opp.append({"card": card, "result": opp_result})

    my_stats_desc = _card_stats_from_rows(rows_my, min_games=min_games, sort_desc=True)
    my_stats_asc = list(reversed(my_stats_desc))

    opp_stats_desc = _card_stats_from_rows(
        rows_opp, min_games=min_games, sort_desc=True
    )
    opp_stats_asc = list(reversed(opp_stats_desc))

    return {
        "best_cards": my_stats_desc,
        "worst_cards": my_stats_asc,
        "tough_opp_cards": opp_stats_desc,
        "easy_opp_cards": opp_stats_asc,
    }


# ---------- Deck-level stats (exact deck lists) ----------


def compute_deck_performance(
    battles_normalized: List[Dict[str, Any]], min_games: int = 3
) -> Dict[str, Any]:
    """
    Compute performance by my deck and opponent deck.

    Deck key = sorted tuple of 8 card names.
    """
    my_decks: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    )
    opp_decks: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    )

    for b in battles_normalized:
        result = b["result"]
        my_key = tuple(sorted(b.get("my_cards", [])))
        opp_key = tuple(sorted(b.get("opp_cards", [])))

        ms = my_decks[my_key]
        ms["games"] += 1
        if result == "win":
            ms["wins"] += 1
        elif result == "loss":
            ms["losses"] += 1
        else:
            ms["draws"] += 1

        os = opp_decks[opp_key]
        os["games"] += 1
        if result == "win":
            os["losses"] += 1
        elif result == "loss":
            os["wins"] += 1
        else:
            os["draws"] += 1

    def _deck_dicts(
        decks_stats: Dict[Tuple[str, ...], Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for deck_key, s in decks_stats.items():
            if s["games"] < min_games:
                continue
            wr = s["wins"] / s["games"] if s["games"] > 0 else 0.0
            out.append(
                {
                    "deck": list(deck_key),
                    "games": s["games"],
                    "wins": s["wins"],
                    "losses": s["losses"],
                    "draws": s["draws"],
                    "win_rate": wr,
                }
            )
        out.sort(key=lambda x: (x["win_rate"], x["games"]), reverse=True)
        return out

    my_decks_list = _deck_dicts(my_decks)
    opp_decks_list = _deck_dicts(opp_decks)

    return {
        "best_decks": my_decks_list,
        "worst_decks": list(reversed(my_decks_list)),
        "tough_matchups": opp_decks_list,
        "easy_matchups": list(reversed(opp_decks_list)),
    }


# ---------- Main entrypoint ----------


def compute_user_analytics(
    battles_normalized: List[Dict[str, Any]],
    min_card_games: int = 3,
    min_deck_games: int = 3,
) -> Dict[str, Any]:
    """
    Main entrypoint for user analytics (and also meta analytics).

    Returns dict:

        {
          "summary": {...},
          "best_cards": [...],
          "worst_cards": [...],
          "tough_opp_cards": [...],
          "easy_opp_cards": [...],
          "best_decks": [...],
          "worst_decks": [...],
          "tough_matchups": [...],         # deck-level (user vs specific decks)
          "easy_matchups": [...],          # deck-level (user vs specific decks)
          "my_deck_types": [...],          # aggregate by archetype (you)
          "opp_deck_types": [...],         # aggregate by archetype (opponents)
          "deck_type_matchups": [...],     # NEW: my_type vs opp_type stats
          "plots": {...}
        }
    """
    df = build_battles_dataframe(battles_normalized)

    summary = compute_summary(df)
    card_stats = compute_card_performance(df, min_games=min_card_games)
    deck_stats = compute_deck_performance(
        battles_normalized, min_games=min_deck_games
    )

    my_deck_types, opp_deck_types = summarize_deck_types(battles_normalized)

    # Deck-level matchups from *your* perspective (kept for potential future use)
    overall_wr = summary.get("win_rate", 0.0)
    user_tough_matchups, user_easy_matchups = compute_user_deck_matchups(
        battles_normalized,
        overall_win_rate=overall_wr,
        min_games=1,
        winrate_delta=0.0,
    )

    # NEW: deck-type vs deck-type matrix (what Phase 2 cares about)
    deck_type_matchups = compute_deck_type_matchups(
        battles_normalized,
        min_games=1,  # you can bump this to 2–3 later if you want to filter tiny samples
    )

    analytics: Dict[str, Any] = {
        "summary": summary,
        "best_cards": card_stats["best_cards"],
        "worst_cards": card_stats["worst_cards"],
        "tough_opp_cards": card_stats["tough_opp_cards"],
        "easy_opp_cards": card_stats["easy_opp_cards"],
        "best_decks": deck_stats["best_decks"],
        "worst_decks": deck_stats["worst_decks"],
        # keep deck-level matchups (not used by USER_MATCHUP_SUMMARY anymore)
        "tough_matchups": user_tough_matchups,
        "easy_matchups": user_easy_matchups,
        "my_deck_types": my_deck_types,
        "opp_deck_types": opp_deck_types,
        "deck_type_matchups": deck_type_matchups,  # <--- NEW KEY
        "plots": {},
    }

    return analytics
