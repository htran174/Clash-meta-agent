# src/analytics/meta_llm_tables.py

from __future__ import annotations

from typing import Any, Dict, List


def build_meta_deck_summary(
    meta_table: List[Dict[str, Any]],
    *,
    min_games_per_type: int = 50,
) -> List[Dict[str, Any]]:
    """
    Aggregate a participant-level meta_table into one compact row per deck type.

    meta_table rows are expected to look like:
        {
          "deck_type": "Cycle",
          "result": "win" | "loss" | "draw",
          ...
        }

    Returns a list of rows:
        {
          "deck_type": str,
          "games": int,
          "meta_share": float,   # fraction of total games (0-1)
          "wins": int,
          "losses": int,
          "draws": int,
          "win_rate": float,     
          "sample_ok": bool,     # games >= min_games_per_type
        }
    """
    stats: Dict[str, Dict[str, Any]] = {}

    # First pass: count games / results per archetype
    for row in meta_table:
        deck_type = row.get("deck_type") or "Unknown"
        result = row.get("result")

        rec = stats.setdefault(
            deck_type,
            {
                "deck_type": deck_type,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "meta_share": 0.0,
                "win_rate": 0.0,
                "sample_ok": False,
            },
        )

        rec["games"] += 1
        if result == "win":
            rec["wins"] += 1
        elif result == "loss":
            rec["losses"] += 1
        elif result == "draw":
            rec["draws"] += 1

    if not stats:
        return []

    # Second pass: compute meta_share, win_rate, sample_ok
    total_games = sum(rec["games"] for rec in stats.values()) or 1
    for rec in stats.values():
        games = rec["games"] or 0
        rec["meta_share"] = games / total_games if total_games > 0 else 0.0
        rec["win_rate"] = rec["wins"] / games if games > 0 else 0.0
        rec["sample_ok"] = games >= min_games_per_type

    # Return sorted by games (most common archetypes first)
    return sorted(stats.values(), key=lambda r: r["games"], reverse=True)


def _label_advantage(
    win_rate: float,
    *,
    neutral: float = 0.5,
    margin: float = 0.05,
) -> str:
    """Map a win rate to 'favored' / 'even' / 'unfavored'."""
    if win_rate >= neutral + margin:
        return "favored"
    if win_rate <= neutral - margin:
        return "unfavored"
    return "even"


def build_meta_matchup_summary(
    matchups: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    min_matchup_games: int = 30,
) -> List[Dict[str, Any]]:
    """
    Flatten the deck_type_matchups matrix from meta_analytics into a
    compact LLM-friendly table.

    Input: matchups[attacker_type][defender_type] = {
               "games": int,
               "wins": int,
               "losses": int,
               "draws": int,
               "win_rate": float,
           }

    Output: list of rows:
        {
          "attacker_type": str,
          "defender_type": str,
          "games": int,
          "wins": int,
          "losses": int,
          "draws": int,
          "win_rate": float,
          "advantage_label": "favored" | "even" | "unfavored",
        }

    Rows with games < min_matchup_games are dropped.
    """
    rows: List[Dict[str, Any]] = []

    for attacker_type, vs_dict in matchups.items():
        if not isinstance(vs_dict, dict):
            continue
        for defender_type, cell in vs_dict.items():
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
                    "attacker_type": attacker_type,
                    "defender_type": defender_type,
                    "games": games,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "win_rate": win_rate,
                    "advantage_label": _label_advantage(win_rate),
                }
            )

    # Sort by games descending so the most representative matchups appear first
    rows.sort(key=lambda r: r["games"], reverse=True)
    return rows
