# src/analytics/user_llm_tables.py

from __future__ import annotations

from typing import Any, Dict, List


def build_user_deck_summary(
    analytics: Dict[str, Any],
    *,
    min_games_per_deck: int = 20,
) -> List[Dict[str, Any]]:
    """
    Build a compact, LLM-friendly per-deck summary for a single user.

    Expects user_analytics to contain:
        analytics["deck_type_matchups"][my_type][opp_type] = {
            "games": int,
            "wins": int,
            "losses": int,
            "draws": int,
            "win_rate": float,
        }

    We aggregate over all opponents for each of the user's deck types.

    Returns list of rows like:
        {
          "deck_type": str,
          "games": int,
          "user_share": float,  # fraction of user's games (0-1)
          "wins": int,
          "losses": int,
          "draws": int,
          "win_rate": float,    
          "sample_ok": bool,    # games >= min_games_per_deck
        }
    """
    matchups = analytics.get("deck_type_matchups", {}) or {}
    if not isinstance(matchups, dict):
        return []

    stats: Dict[str, Dict[str, Any]] = {}

    # Aggregate over all opponents for each "my" deck type
    for my_type, vs_dict in matchups.items():
        if not isinstance(vs_dict, dict):
            continue

        rec = stats.setdefault(
            my_type,
            {
                "deck_type": my_type,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "user_share": 0.0,
                "win_rate": 0.0,
                "sample_ok": False,
            },
        )

        for _opp_type, cell in vs_dict.items():
            if not isinstance(cell, dict):
                continue
            rec["games"] += int(cell.get("games", 0))
            rec["wins"] += int(cell.get("wins", 0))
            rec["losses"] += int(cell.get("losses", 0))
            rec["draws"] += int(cell.get("draws", 0))

    if not stats:
        return []

    # Compute totals and derived metrics
    total_games = sum(rec["games"] for rec in stats.values()) or 1
    for rec in stats.values():
        games = rec["games"] or 0
        rec["user_share"] = games / total_games if total_games > 0 else 0.0
        rec["win_rate"] = rec["wins"] / games if games > 0 else 0.0
        rec["sample_ok"] = games >= min_games_per_deck

    # Sort most-played decks first
    return sorted(stats.values(), key=lambda r: r["games"], reverse=True)


def _label_advantage(
    win_rate: float,
    *,
    neutral: float = 0.5,
    margin: float = 0.05,
) -> str:
    """
    Map a win rate into 'favored' / 'even' / 'unfavored' from the user's POV.
    """
    if win_rate >= neutral + margin:
        return "favored"
    if win_rate <= neutral - margin:
        return "unfavored"
    return "even"


def build_user_matchup_summary(
    analytics: Dict[str, Any],
    *,
    min_matchup_games: int = 10,
) -> List[Dict[str, Any]]:
    """
    Flatten the user's deck_type_matchups into a compact matchup table.

    Input (from user_analytics):
        analytics["deck_type_matchups"][my_type][opp_type] = {
            "games": int,
            "wins": int,
            "losses": int,
            "draws": int,
            "win_rate": float,
        }

    Returns list of rows like:
        {
          "my_deck_type": str,
          "opp_deck_type": str,
          "games": int,
          "wins": int,
          "losses": int,
          "draws": int,
          "win_rate": float,        # from user POV
          "advantage_label": str,   # 'favored' | 'even' | 'unfavored'
        }

    Rows with too few games are dropped (min_matchup_games).
    """
    matchups = analytics.get("deck_type_matchups", {}) or {}
    if not isinstance(matchups, dict):
        return []

    rows: List[Dict[str, Any]] = []

    for my_type, vs_dict in matchups.items():
        if not isinstance(vs_dict, dict):
            continue

        for opp_type, cell in vs_dict.items():
            if not isinstance(cell, dict):
                continue

            games = int(cell.get("games", 0))
            if games < min_matchup_games:
                continue

            wins = int(cell.get("wins", 0))
            losses = int(cell.get("losses", 0))
            draws = int(cell.get("draws", 0))
            win_rate = float(cell.get("win_rate", 0.0))

            rows.append(
                {
                    "my_deck_type": my_type,
                    "opp_deck_type": opp_type,
                    "games": games,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "win_rate": win_rate,
                    "advantage_label": _label_advantage(win_rate),
                }
            )

    # Sort by games so the most meaningful matchups appear first
    rows.sort(key=lambda r: r["games"], reverse=True)
    return rows
