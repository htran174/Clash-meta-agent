from typing import List, Dict, Any

from .cr_client import cr_get


def normalize_player_tag(tag: str) -> str:
    """
    Normalize a Clash Royale player tag:

    - Strip whitespace
    - Ensure it starts with '#'
    - Uppercase (tags are case-insensitive but usually uppercase)
    """
    cleaned = tag.strip().upper()
    if not cleaned.startswith("#"):
        cleaned = "#" + cleaned
    return cleaned


def get_player_battlelog(player_tag: str) -> List[Dict[str, Any]]:
    """
    Fetch the battlelog for a given player tag.

    Uses:
      GET /v1/players/%23{TAG}/battlelog

    Args:
        player_tag: Player tag, with or without leading '#'.

    Returns:
        List of battle dicts (raw API format).
    """
    normalized = normalize_player_tag(player_tag)
    # API expects '#' encoded as '%23'
    encoded_tag = normalized.replace("#", "%23", 1)

    data = cr_get(f"/players/{encoded_tag}/battlelog")

    #this endpoint returns a JSON array (list of battles)
    if isinstance(data, list):
        return data

    # Fallback in case the response is wrapped
    return data.get("items", [])
