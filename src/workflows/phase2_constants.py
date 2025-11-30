# src/workflows/phase2_constants.py
# Constants + Enums for Phase 2 (Q&A Routing)

# --------------------------------------------
# MODEL CHOICES FOR PHASE 2
# --------------------------------------------
# Cheap classifier model
CLASSIFIER_MODEL = "gpt-4.1-nano"

# Stronger coach model
EXPERT_MODEL = "gpt-4.1-mini"


# --------------------------------------------
# ALLOWED QUESTION CATEGORIES
# --------------------------------------------
CATEGORIES = [
    "user",       # questions about user's own performance / playstyle
    "matchup",    # losing to certain decks, counters, WR vs X
    "meta",       # global meta questions
    "card",       # card usage, best cards, cards user loses to
    "other",      # unclear / fallback
]

# --------------------------------------------
# ATOMIC DATA BLOCKS (Phase 2 DataNeeds)
# --------------------------------------------
DATANEEDS = [
    "USER_SUMMARY",          # user overall stats
    "USER_DECK_SUMMARY",     # decks user plays + winrates
    "USER_MATCHUP_SUMMARY",  # user's WR vs deck types

    "USER_CARD_SUMMARY",     # user performance with cards
    "OPPONENT_CARD_SUMMARY", # cards the user struggles vs

    "META_DECK_SUMMARY",     # meta deck types + performance
    "META_DECK_MATCHUPS",    # meta-level deck-vs-deck performance

    "SEND_ALL",              # fallback
]

# --------------------------------------------
# DEFAULT DATA NEEDS for each category
# (used ONLY if LLM output is empty or invalid)
# --------------------------------------------
DEFAULT_NEEDS = {
    "user": ["USER_SUMMARY", "USER_DECK_SUMMARY"],
    "matchup": ["USER_MATCHUP_SUMMARY"],
    "meta": ["META_DECK_SUMMARY"],
    "card": ["USER_CARD_SUMMARY"],
    "other": ["SEND_ALL"],
}

# --------------------------------------------
# CLASSIFIER SYSTEM PROMPT
#Goal: only get need data to save token
# --------------------------------------------
CLASSIFIER_SYSTEM_PROMPT = """
You are a routing classifier for a Clash Royale analytics assistant.

Your job:
1. Classify the user's question into ONE category.
2. Choose ONLY the minimal data blocks ("data_needs") required to answer it.

Allowed categories:
- "user"      → performance or decks the user plays.
- "matchup"   → losing to certain deck types, counters, deck-vs-deck questions.
- "meta"      → global meta, popular decks, strongest archetypes.
- "card"      → user's good/bad cards, or cards they lose to.
- "other"     → anything unrelated or unclear.

Allowed data_needs (Atomic building blocks):
- "USER_SUMMARY"
- "USER_DECK_SUMMARY"
- "USER_MATCHUP_SUMMARY"
- "USER_CARD_SUMMARY"
- "OPPONENT_CARD_SUMMARY"
- "META_DECK_SUMMARY"
- "META_DECK_MATCHUPS"
- "SEND_ALL"

Short meanings:
- USER_SUMMARY: overall user performance metrics
- USER_DECK_SUMMARY: user's deck types + winrates
- USER_MATCHUP_SUMMARY: user's performance vs deck types
- USER_CARD_SUMMARY: user performance with cards
- OPPONENT_CARD_SUMMARY: cards the user struggles against
- META_DECK_SUMMARY: meta deck types + their winrates
- META_DECK_MATCHUPS: how meta decks perform vs each other
- SEND_ALL: fallback

Rules:
- You may choose MULTIPLE data blocks.
- Always choose the MINIMAL blocks needed.
- If the question asks:
    - "Why do I lose to X?" → include USER_MATCHUP_SUMMARY
    - "What counters X in meta?" → include META_DECK_MATCHUPS
    - "What cards am I losing to?" → include OPPONENT_CARD_SUMMARY
    - "What are my best cards?" → include USER_CARD_SUMMARY
    - "How am I doing?" → include USER_SUMMARY
    - "What deck do I win with?" → include USER_DECK_SUMMARY
    - "What's strong in meta?" → include META_DECK_SUMMARY
- If unclear → category="other", data_needs=["SEND_ALL"]

Return ONLY valid JSON:
{"category": "...", "data_needs": ["...", "..."]}
"""