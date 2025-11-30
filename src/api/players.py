# src/api/players.py
from typing import List, Dict, Any
from .cr_client import get_global_top_players


def fetch_top_players(limit = 300) -> List[Dict[str, Any]]:
    """
    Returns a list of the top ~300 (if not specifi) global ladder players
    from the leaderboard endpoint.
    """
    data = get_global_top_players(limit)

    if not data:
        print("ERROR: get_global_top_players() returned empty response.")
        return []

    items = data.get("items", [])
    if not items:
        print("WARNING: No players found in 'items'. Full response:")
        print(data)
        return []

    return items
