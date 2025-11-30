#src/analytics/meta_builder.py
from typing import Any, Dict, List

from src.api.players import fetch_top_300_players
from src.api.battles import get_player_battlelog
from src.analytics.battle_filters import filter_and_normalize_ranked_1v1
from src.analytics.user_analytics import compute_user_analytics
from src.analytics.plots import generate_card_plots
from src.utils.sampling import sample_players


def build_meta_analytics(
    *,
    max_players: int = 300,
    sample_size: int = 50,
    per_player_matches: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build the Phase 0 meta_analytics dataset.

    Steps (per spec):
      1) Fetch top ~300 global players.
      2) Randomly sample 50 players.
      3) For each sampled player:
           - Fetch battlelog.
           - Filter to ranked/Trophy Road 1v1.
           - Take last 10 valid matches.
      4) Combine into meta_battles (~500 battles).
      5) Compute analytics using same engine as user_analytics.
      6) Generate plots with prefix "meta".

    Returns:
        meta_analytics dict with the same schema as user_analytics.
    """
    if verbose:
        print("Fetching top global players...")

    top_players = fetch_top_300_players()
    if max_players and len(top_players) > max_players:
        top_players = top_players[:max_players]

    if verbose:
        print(f"Total players fetched: {len(top_players)}")
        print(f"Sampling {sample_size} players...")

    sampled_players = sample_players(top_players, sample_size=sample_size)

    meta_battles: List[Dict[str, Any]] = []

    for idx, player in enumerate(sampled_players, start=1):
        tag = player.get("tag")
        name = player.get("name", "")

        if not tag:
            if verbose:
                print(f"[{idx}/{len(sampled_players)}] Skipping player with no tag.")
            continue

        if verbose:
            print(f"[{idx}/{len(sampled_players)}] Fetching battles for {name} ({tag})...")

        try:
            raw_battles = get_player_battlelog(tag)
            ranked_normalized = filter_and_normalize_ranked_1v1(raw_battles)

            # Take the last N ranked matches (API returns most recent first)
            player_slice = ranked_normalized[:per_player_matches]
            meta_battles.extend(player_slice)

            if verbose:
                print(
                    f"  -> {len(raw_battles)} raw battles, "
                    f"{len(ranked_normalized)} ranked, "
                    f"using {len(player_slice)} for meta."
                )

        except Exception as e:
            if verbose:
                print(f"  !! Error fetching/processing player {tag}: {e}")
            continue

    if verbose:
        print(f"\nTotal meta battles collected: {len(meta_battles)}")
        print("Computing meta_analytics...")

    meta_analytics = compute_user_analytics(meta_battles)
    meta_analytics = generate_card_plots(meta_analytics, prefix="meta")

    if verbose:
        print("meta_analytics ready.")
        print("Summary:", meta_analytics.get("summary", {}))

    return meta_analytics
