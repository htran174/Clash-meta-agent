#src/analytics/plots.py
import os
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

PLOTS_DIR = "plots"


def _ensure_plots_dir() -> None:
    """Make sure the plots directory exists."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _top_n_cards(cards: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    """Return top-n entries from a card stats list."""
    return cards[:n]


def plot_card_bar_chart(
    cards: List[Dict[str, Any]],
    title: str,
    filename: str,
    *,
    metric: str = "win_rate",
) -> str:
    """
    Generic bar chart for card stats.

    Args:
        cards: List of card dicts with at least keys: "card", metric.
        title: Plot title.
        filename: File name (inside plots/).
        metric: Metric key to plot on y-axis (default: "win_rate").

    Returns:
        Relative path to the saved PNG.
    """
    _ensure_plots_dir()

    if not cards:
        return os.path.join(PLOTS_DIR, f"{filename}.png")

    top_cards = _top_n_cards(cards, n=10)
    labels = [c["card"] for c in top_cards]
    values = [c.get(metric, 0.0) for c in top_cards]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{filename}.png")
    fig.savefig(path)
    plt.close(fig)

    return path


def plot_deck_type_pie(
    deck_types: List[Dict[str, Any]],
    title: str,
    filename: str,
) -> str:
    """
    Pie chart for deck types (by games played).

    deck_types entries look like:
        {
          "type": "Beatdown",
          "games": 12,
          "wins": ...,
          "losses": ...,
          "draws": ...,
          "win_rate": 0.58,
        }
    """
    _ensure_plots_dir()

    if not deck_types:
        return os.path.join(PLOTS_DIR, f"{filename}.png")

    labels = [d["type"] for d in deck_types]
    sizes = [d["games"] for d in deck_types]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax.set_title(title)

    path = os.path.join(PLOTS_DIR, f"{filename}.png")
    fig.savefig(path)
    plt.close(fig)

    return path

def plot_deck_type_bar(
    deck_types: List[Dict[str, Any]],
    title: str,
    filename: str,
    *,
    metric: str = "win_rate",
) -> str:
    """
    Bar chart for deck types (e.g. my win rate vs each opponent deck type).

    deck_types entries look like:
        {
          "type": "Beatdown",
          "games": 12,
          "wins": ...,
          "losses": ...,
          "draws": ...,
          "win_rate": 0.58,
        }
    """
    _ensure_plots_dir()

    if not deck_types:
        return os.path.join(PLOTS_DIR, f"{filename}.png")

    labels = [d["type"] for d in deck_types]
    values = [d.get(metric, 0.0) for d in deck_types]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{filename}.png")
    fig.savefig(path)
    plt.close(fig)

    return path


def generate_card_plots(analytics: Dict[str, Any], prefix: str = "user") -> Dict[str, Any]:
    """
    Generate plots for the given analytics dict and
    update analytics["plots"] with file paths.

    Currently creates:
      - best_cards, worst_cards (bar charts)
      - tough_opp_cards, easy_opp_cards (bar charts)
      - my_deck_types (pie)
      - opp_deck_types (pie)
    """
    plots = analytics.get("plots", {})

    best_cards = analytics.get("best_cards", [])
    worst_cards = analytics.get("worst_cards", [])
    tough_opp_cards = analytics.get("tough_opp_cards", [])
    easy_opp_cards = analytics.get("easy_opp_cards", [])

    my_deck_types = analytics.get("my_deck_types", [])
    opp_deck_types = analytics.get("opp_deck_types", [])

    # Card bar charts
    plots["best_cards"] = plot_card_bar_chart(
        best_cards,
        title="Best Cards (Win Rate)",
        filename=f"{prefix}_best_cards",
    )

    plots["worst_cards"] = plot_card_bar_chart(
        worst_cards,
        title="Worst Cards (Win Rate)",
        filename=f"{prefix}_worst_cards",
    )

    plots["tough_opp_cards"] = plot_card_bar_chart(
        tough_opp_cards,
        title="Opponent Threat Cards (Their Win Rate)",
        filename=f"{prefix}_tough_opp_cards",
    )

    plots["easy_opp_cards"] = plot_card_bar_chart(
        easy_opp_cards,
        title="Opponent Easy Cards (Their Win Rate)",
        filename=f"{prefix}_easy_opp_cards",
    )

    # Deck-type pie charts
    plots["my_deck_types_pie"] = plot_deck_type_pie(
        my_deck_types,
        title="My Deck Types (by Games)",
        filename=f"{prefix}_my_deck_types",
    )

    plots["opp_deck_types_pie"] = plot_deck_type_pie(
        opp_deck_types,
        title="Opponent Deck Types (by Games)",
        filename=f"{prefix}_opp_deck_types",
    )
    
    plots["opp_deck_types_bar"] = plot_deck_type_bar(
        opp_deck_types,
        title="My Win Rate vs Opponent Deck Types",
        filename=f"{prefix}_opp_deck_types_bar",
    )

    analytics["plots"] = plots
    return analytics
