# src/api/cr_client.py
import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

CR_API_KEY = os.getenv("CR_API_KEY")
BASE_URL = "https://api.clashroyale.com/v1"


def _get_headers() -> Dict[str, str]:
    """Return auth headers for the Clash Royale API."""
    if not CR_API_KEY:
        raise RuntimeError(
            "CR_API_KEY is not set. Please add it to your .env file."
        )
    return {"Authorization": f"Bearer {CR_API_KEY}"}


def cr_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Low-level helper for GET requests to the Clash Royale API.

    Args:
        path: API path starting with '/v1/...'
        params: Optional query parameters.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        RuntimeError if the response status code is not 200.
    """
    url = f"{BASE_URL}{path}"
    response = requests.get(url, headers=_get_headers(), params=params, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(
            f"Clash Royale API error {response.status_code}: {response.text}"
        )

    return response.json()

LEADERBOARD_GLOBAL_ID = 170000005  #Rank 1v1 ( not trophi road)


def get_global_top_players(limit: int = 300) -> Dict[str, Any]:
    """
    Fetch players from the global ladder leaderboard.

    Wraps:
        GET /leaderboard/{leaderboardId}?limit=...

    leaderboardId 170000005 = global trophy leaderboard (per CR docs).
    """
    params: Dict[str, Any] = {"limit": limit}
    path = f"/leaderboard/{LEADERBOARD_GLOBAL_ID}"
    return cr_get(path, params=params)

