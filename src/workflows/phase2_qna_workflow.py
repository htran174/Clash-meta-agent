#qna workflow
import json
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from openai import OpenAI
from src.api.llm_client import chat_completion 

from src.workflows.phase2_constants import CATEGORIES, DATANEEDS, DEFAULT_NEEDS, CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_MODEL, EXPERT_MODEL

from src.workflows.meta_workflow import build_meta_graph
from src.workflows.user_workflow import build_user_analytics_graph

#---------Class--------------
class QnAState(TypedDict, total=False):
    # Inputs
    user_tag: str
    question: str

    user_analytics: Dict[str, Any]
    user_llm_tables: Dict[str, Any]
    meta_analytics: Dict[str, Any]
    meta_llm_tables: Dict[str, Any]
    meta_table: Any

    # Classification
    question_category: str
    question_data_needs: List[str]

    # Data health check
    games_played: int
    has_enough_data: bool
    low_data_warning: str

    # Context for expert model
    context_text: str          # human-readable summary
    context_tables: Dict[str, Any]  # raw tables (user + meta slices)

    # Final answer
    answer: str

    # Debug log
    notes: List[str]


#---------------Helper---------------------
def build_classifier_user_prompt(question: str) -> str:
    return f"""
Classify the following question. Respond in JSON only.

User question:
\"\"\"{question}\"\"\"
"""
def prep_user_context_node(state):
    notes = state.setdefault("notes", [])
    data_needs = state.get("question_data_needs", [])

    user_llm = state.get("user_llm_tables", {})
    summary_table = user_llm.get("user_summary", [])
    deck_table = user_llm.get("user_deck_summary", [])

    context_tables = {}
    summary_lines = []

    # Include USER_SUMMARY
    if "USER_SUMMARY" in data_needs:
        context_tables["user_summary"] = summary_table

        # Convert summary into readable lines
        for row in summary_table:
            summary_lines.append(f"{row['metric']}: {row['value']}")

    # Include USER_DECK_SUMMARY
    if "USER_DECK_SUMMARY" in data_needs:
        context_tables["user_deck_summary"] = deck_table

        if deck_table:
            summary_lines.append("\nYour deck performance:")
            for row in deck_table[:5]:
                summary_lines.append(
                    f"- {row['deck_type']}: {row['wins']}/{row['games']} wins "
                    f"({row['win_rate']:.2f})"
                )
        else:
            summary_lines.append("\nNo deck statistics found for this user.")

    state["context_tables"] = context_tables
    state["context_text"] = "\n".join(summary_lines)
    notes.append("prep_user_context: done")
    return state

def prep_matchup_context_node(state: QnAState) -> QnAState:
    """
    Prepare context for matchup-style questions.

    Always include:
      - user_summary (overall stats)
      - user_matchup_summary (can be empty)
      - meta_deck_summary
      - meta_matchup_summary
    """
    notes = state.setdefault("notes", [])
    needs = state.get("question_data_needs", []) or []

    user_llm = state.get("user_llm_tables", {}) or {}
    meta_llm = state.get("meta_llm_tables", {}) or {}

    user_summary = user_llm.get("user_summary", [])
    user_matchups = user_llm.get("user_matchup_summary", [])
    meta_decks = meta_llm.get("meta_deck_summary", [])
    meta_matchups = meta_llm.get("meta_matchup_summary", [])

    summary_lines: list[str] = []

    # User overall stats
    if user_summary:
        summary_lines.append("User overall stats are included (games, wins, losses, win rate).")
    else:
        summary_lines.append("User overall stats are missing.")

    # User matchup stats
    if user_matchups:
        summary_lines.append(
            f"User has matchup stats vs {len(user_matchups)} archetypes "
            "showing win/loss rates by deck type."
        )
    else:
        summary_lines.append(
            "User has no recorded matchup rows yet; you'll need to lean more on global meta patterns "
            "and general coaching, not exact per-archetype winrates."
        )

    # Meta stats
    if meta_decks:
        summary_lines.append(
            f"Meta deck summary is available for {len(meta_decks)} archetypes "
            "(games, win rate, meta share)."
        )
    if meta_matchups:
        summary_lines.append(
            f"Meta matchup summary is available for {len(meta_matchups)} archetype pairs "
            "(attacker vs defender win rates)."
        )

    state["context_text"] = "\n".join(summary_lines)

    # Always ship these tables for the expert model
    state["context_tables"] = {
        "user_summary": user_summary,
        "user_matchup_summary": user_matchups,
        "meta_deck_summary": meta_decks,
        "meta_matchup_summary": meta_matchups,
    }

    notes.append("prep_matchup_context: done")
    return state


def prep_card_context_node(state):
    notes = state.setdefault("notes", [])
    data_needs = state.get("question_data_needs", [])

    user_llm = state.get("user_llm_tables", {})

    user_cards = user_llm.get("user_card_summary", [])
    opp_cards = user_llm.get("opponent_card_summary", [])

    context_tables = {}
    summary_lines = []

    if "USER_CARD_SUMMARY" in data_needs:
        context_tables["user_card_summary"] = user_cards
        summary_lines.append("Your card performance is included.")

    if "OPPONENT_CARD_SUMMARY" in data_needs:
        context_tables["opponent_card_summary"] = opp_cards
        summary_lines.append("Opponent card performance included.")

    state["context_tables"] = context_tables
    state["context_text"] = "\n".join(summary_lines)
    notes.append("prep_card_context: done")
    return state

def prep_meta_context_node(state):
    notes = state.setdefault("notes", [])
    data_needs = state.get("question_data_needs", [])

    meta_llm = state.get("meta_llm_tables", {})

    meta_decks = meta_llm.get("meta_deck_summary", [])
    meta_matchups = meta_llm.get("meta_matchup_summary", [])

    context_tables = {}
    summary_lines = []

    if "META_DECK_SUMMARY" in data_needs:
        context_tables["meta_deck_summary"] = meta_decks
        summary_lines.append("Meta deck performance table included.")

    if "META_DECK_MATCHUPS" in data_needs:
        context_tables["meta_deck_matchups"] = meta_matchups
        summary_lines.append("Meta deck matchup chart included.")

    state["context_text"] = "\n".join(summary_lines)
    state["context_tables"] = context_tables
    notes.append("prep_meta_context: done")
    return state

def prep_other_context_node(state):
    notes = state.setdefault("notes", [])

    # Just provide everything we can
    summary = state.get("user_llm_tables", {}).get("user_summary", [])

    context_tables = {"user_summary": summary}

    state["context_tables"] = context_tables
    state["context_text"] = "General summary included."
    notes.append("prep_other_context: done")
    return state


#----------------nodes----------------------------
def classify_question_node(state):
    question = state.get("question", "")
    notes = state.setdefault("notes", [])

    user_prompt = build_classifier_user_prompt(question)

    # Call cheap classifier model through api.llm_client
    try:
        raw = chat_completion(
            model=CLASSIFIER_MODEL,
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=300,
        )
    except Exception as e:
        notes.append(f"classify_question: LLM error → fallback ({e})")
        state["question_category"] = "other"
        state["question_data_needs"] = ["SEND_ALL"]
        return state

    # Parse JSON
    try:
        parsed = json.loads(raw)
        category = parsed.get("category")
        needs = parsed.get("data_needs", [])
    except Exception:
        notes.append("classify_question: JSON parse error → fallback")
        category = "other"
        needs = ["SEND_ALL"]

    if category not in CATEGORIES:
        notes.append(f"classify_question: invalid category '{category}' → fallback")
        category = "other"
        needs = ["SEND_ALL"]

    # Validate / filter data_needs
    cleaned = [n for n in needs if n in DATANEEDS]
    if not cleaned:
        cleaned = DEFAULT_NEEDS[category]

    state["question_category"] = category
    state["question_data_needs"] = cleaned
    notes.append(f"classify_question: {category} → {cleaned}")
    return state

def start_question_node(state: QnAState) -> QnAState:
    """
    Initialize Phase 2 state:
    - ensure notes list exists
    - compute games_played from user_analytics.summary
    """
    notes = state.setdefault("notes", [])

    user_analytics = state.get("user_analytics", {}) or {}
    summary = user_analytics.get("summary", {}) or {}

    games_played = int(summary.get("games_played", 0))
    state["games_played"] = games_played

    # Ensure question field exists
    state.setdefault("question", "")

    notes.append(f"start_question: games_played={games_played}")
    return state

def enough_data_node(state: QnAState) -> QnAState:
    """
    Simple data health check BEFORE the expert LLM.

    - For non-meta questions, we look at the user's games_played and add a small warning if < 20.
    - For meta questions, we ALWAYS treat data as sufficient and skip the user-sample warning,
      because meta is based on Phase 0 (large sample) not on the user's games.
    """
    notes = state.setdefault("notes", [])
    games = state.get("games_played", 0) or 0
    category = state.get("question_category", "other")

    # Meta-only questions: no user-sample warning
    if category == "meta":
        state["has_enough_data"] = True
        state["low_data_warning"] = ""
        notes.append(
            f"enough_data: category=meta → skipping user-sample warning "
            f"(games={games})"
        )
        return state

    # Non-meta questions: keep the simple threshold
    if games >= 20:
        state["has_enough_data"] = True
        state["low_data_warning"] = ""
    else:
        state["has_enough_data"] = False
        state["low_data_warning"] = (
            f"Warning: only {games} recent ranked games; "
            "these stats may be noisy or not fully representative."
        )

    notes.append(
        f"enough_data: category={category}, games={games}, "
        f"has_enough_data={state['has_enough_data']}"
    )
    return state

def expert_answer_llm_node(state: QnAState) -> QnAState:
    """
    Expensive expert LLM call (gpt-4.1-mini).
    Uses context_text + context_tables + low_data_warning.
    Now prints everything we send for easy debugging.
    """
    import json

    notes = state.setdefault("notes", [])

    question = state.get("question", "")
    category = state.get("question_category", "other")
    low_data_warning = state.get("low_data_warning", "")
    context_text = state.get("context_text", "")
    context_tables = state.get("context_tables", {}) or {}

    # ---- DEBUG: show what we're sending ----
    #print("\n==========================")
    #print("EXPERT INPUT DEBUG")
    #print("==========================")

    #print("\n[Question]")
    #print(question)

    #print("\n[Category]")
    #print(category)

    #print("\n[Data quality warning]")
    #print(low_data_warning or "(none)")

    #print("\n[Context text]")
    #print(context_text or "(empty)")

    #print("\n[Context tables summary]")
    #for name, table in context_tables.items():
        #if isinstance(table, list):
            #print(f"- {name}: list with {len(table)} rows")
            #for row in table[:2]:  # show first 2 rows for each table
                #print(f"    sample row: {row}")
        #else:
            #print(f"- {name}: {type(table).__name__}")

    #print("\n==========================\n")

    # ---- Build prompts for the expert model ----

    # System prompt: keep warning small, but still give best-effort advice
    system_prompt = (
        "You are a Clash Royale coach.\n"
        "You receive:\n"
        "- A short text summary of available stats (user + meta).\n"
        "- One or more tables in JSON form (user_summary, user_matchup_summary, meta_deck_summary, etc.).\n"
        "- An optional data quality warning.\n\n"
        "Guidelines:\n"
        "1) Always give a clear, concrete answer to the user's question using whatever data is available.\n"
        "2) If user-specific stats are missing or sparse, lean more on the meta tables and general matchup principles.\n"
        "3) Mention the data quality warning briefly once, but do NOT let it dominate the answer.\n"
        "4) Prefer 2–4 short paragraphs with actionable tips.\n"
        "5) Only say you 'can't tell' something if there is truly zero relevant data anywhere.\n"
    )

    # Convert tables to JSON (truncated for safety)
    try:
        tables_json = json.dumps(context_tables, default=str)
    except Exception:
        tables_json = "{}"

    if len(tables_json) > 4000:
        tables_json = tables_json[:4000] + "\n... [truncated]"

    user_prompt = f"""
User question:
{question}

Question category: {category}

Data quality warning (if any):
{low_data_warning or "None."}

Context summary text:
{context_text or "No summary provided."}

Context tables (JSON):
{tables_json}
"""

    # ---- Call expert LLM ----
    try:
        answer = chat_completion(
            model=EXPERT_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=700,
        )
        notes.append("expert_answer_llm: answered successfully")
    except Exception as e:
        answer = (
            "I ran into an error calling the expert model. "
            f"(internal note: {e})"
        )
        notes.append(f"expert_answer_llm: error {e}")

    state["answer"] = answer
    return state



def route_by_category(state: QnAState) -> str:
    """
    Decide which prep_* node to call based mainly on data_needs,
    not just the high-level category label.
    """
    needs = state.get("question_data_needs", []) or []
    cat = state.get("question_category", "other")

    # If it asks for matchup data → matchup prep
    if any(n in needs for n in ["USER_MATCHUP_SUMMARY", "META_DECK_MATCHUPS"]):
        return "prep_matchup_context"

    # If it asks for card-level data → card prep
    if any(n in needs for n in ["USER_CARD_SUMMARY", "OPPONENT_CARD_SUMMARY"]):
        return "prep_card_context"

    # If it asks for meta-only stuff → meta prep
    if any(n in needs for n in ["META_DECK_SUMMARY"]):
        return "prep_meta_context"

    # If it asks for user summary / deck stats → user prep
    if any(n in needs for n in ["USER_SUMMARY", "USER_DECK_SUMMARY"]):
        return "prep_user_context"

    # Fallback: still use the category label as a hint
    if cat == "user":
        return "prep_user_context"
    if cat == "matchup":
        return "prep_matchup_context"
    if cat == "meta":
        return "prep_meta_context"
    if cat == "card":
        return "prep_card_context"

    return "prep_other_context"


def build_qna_graph():
    """
    Build the Phase 2 Q&A coach graph.

    Flow:
      START
        → start_question
        → classify_question
        → (conditional) one of prep_*_context
        → enough_data
        → expert_answer_llm
        → END
    """
    graph = StateGraph(QnAState)

    # Nodes
    graph.add_node("start_question", start_question_node)
    graph.add_node("classify_question", classify_question_node)

    graph.add_node("prep_user_context", prep_user_context_node)
    graph.add_node("prep_matchup_context", prep_matchup_context_node)
    graph.add_node("prep_card_context", prep_card_context_node)
    graph.add_node("prep_meta_context", prep_meta_context_node)
    graph.add_node("prep_other_context", prep_other_context_node)

    graph.add_node("enough_data", enough_data_node)
    graph.add_node("expert_answer_llm", expert_answer_llm_node)

    # Edges
    graph.add_edge(START, "start_question")
    graph.add_edge("start_question", "classify_question")

    # Conditional routing by category
    graph.add_conditional_edges(
        "classify_question",
        route_by_category,
        {
            "prep_user_context": "prep_user_context",
            "prep_matchup_context": "prep_matchup_context",
            "prep_card_context": "prep_card_context",
            "prep_meta_context": "prep_meta_context",
            "prep_other_context": "prep_other_context",
        },
    )

    # All prep_* nodes converge to enough_data → expert_answer → END
    graph.add_edge("prep_user_context", "enough_data")
    graph.add_edge("prep_matchup_context", "enough_data")
    graph.add_edge("prep_card_context", "enough_data")
    graph.add_edge("prep_meta_context", "enough_data")
    graph.add_edge("prep_other_context", "enough_data")

    graph.add_edge("enough_data", "expert_answer_llm")
    graph.add_edge("expert_answer_llm", END)

    app = graph.compile()
    return app

from pydantic import BaseModel

#--------------------adding phase 0 + phase 1 workflow ---------------------------
class CoachState(TypedDict, total=False):
    # human inputs
    player_tag: str
    question: str

    # phase 0 outputs
    meta_analytics: Dict[str, Any]
    meta_llm_tables: Dict[str, Any]
    meta_table: Any  # whatever type your meta_table is

    # phase 1 outputs
    user_analytics: Dict[str, Any]
    user_llm_tables: Dict[str, Any]

    # phase 2 outputs
    answer: str
    notes: List[str]

_meta_graph = build_meta_graph()
_user_graph = build_user_analytics_graph()
_qna_graph = build_qna_graph() 

def ensure_meta(state: CoachState) -> CoachState:
    # If we already have meta, don't recompute
    if "meta_analytics" in state and "meta_llm_tables" in state:
        return state

    # Mimic your notebook: meta_graph.invoke({}, config={"recursion_limit": 80})
    meta_state = _meta_graph.invoke({}, config={"recursion_limit": 80})

    return {
        **state,
        "meta_analytics": meta_state.get("meta_analytics", {}),
        "meta_llm_tables": meta_state.get("meta_llm_tables", {}),
        "meta_table": meta_state.get("meta_table", []),
    }

def ask_for_tag(state: CoachState) -> CoachState:
    # Only ask once per thread
    if "player_tag" in state:
        return state

    tag = interrupt("Please enter your Clash Royale player tag (without #):")
    # When you resume in Studio, `tag` will be whatever you typed
    return {**state, "player_tag": tag}

def ensure_user(state: CoachState) -> CoachState:
    # If we've already built user analytics for this tag, reuse them
    if "user_analytics" in state and "user_llm_tables" in state:
        return state

    player_tag = state["player_tag"]
    user_state = _user_graph.invoke({"player_tag": player_tag})

    return {
        **state,
        "user_analytics": user_state.get("user_analytics", {}),
        "user_llm_tables": user_state.get("user_llm_tables", {}),
        # (optional) keep tag in sync
        "player_tag": user_state.get("player_tag", player_tag),
    }

def ask_for_question(state: CoachState) -> CoachState:
    question = interrupt("What would you like to ask about your Clash performance?"
                         "(Type 'stop' to finish.)")
    return {**state, "question": question}

def ensure_user(state: CoachState) -> CoachState:
    # If we've already built user analytics for this tag, reuse them
    if "user_analytics" in state and "user_llm_tables" in state:
        return state

    player_tag = state["player_tag"]
    user_state = _user_graph.invoke({"player_tag": player_tag})

    return {
        **state,
        "user_analytics": user_state.get("user_analytics", {}),
        "user_llm_tables": user_state.get("user_llm_tables", {}),
        # (optional) keep tag in sync
        "player_tag": user_state.get("player_tag", player_tag),
    }

def ask_for_question(state: CoachState) -> CoachState:
    question = interrupt("What would you like to ask about your Clash performance?")
    return {**state, "question": question}

def qa_answer(state: CoachState) -> CoachState:
    qna_input = {
        "user_tag": state.get("player_tag"),
        "question": state.get("question"),
        "user_analytics": state.get("user_analytics", {}),
        "user_llm_tables": state.get("user_llm_tables", {}),
        "meta_analytics": state.get("meta_analytics", {}),
        "meta_llm_tables": state.get("meta_llm_tables", {}),
        "meta_table": state.get("meta_table", []),
        "notes": state.get("notes", []),
    }

    qna_state = _qna_graph.invoke(qna_input)

    # Adjust these keys if your phase2 graph returns something slightly different
    answer = qna_state.get("answer", "(No 'answer' key returned)")
    notes = qna_state.get("notes", [])

    return {
        **state,
        "answer": answer,
        "notes": notes,
    }


def route_after_question(state: CoachState) -> str:
    """
    Decide what to do after we ask for a question.
    If the user types 'stop' (or similar), end the graph instead of
    going to qa_answer.
    """
    q = (state.get("question") or "").strip().lower()
    if q in {"stop", "exit", "quit"}:
        # Special label we'll map to END in build_coach_graph
        return "end"
    return "qa_answer"



def build_coach_graph():
    graph = StateGraph(CoachState)

    graph.add_node("ensure_meta", ensure_meta)
    graph.add_node("ask_for_tag", ask_for_tag)
    graph.add_node("ensure_user", ensure_user)
    graph.add_node("ask_for_question", ask_for_question)
    graph.add_node("qa_answer", qa_answer)

    graph.add_edge(START, "ensure_meta")
    graph.add_edge("ensure_meta", "ask_for_tag")
    graph.add_edge("ask_for_tag", "ensure_user")
    graph.add_edge("ensure_user", "ask_for_question")

    # NEW: conditional routing after asking the question
    graph.add_conditional_edges(
        "ask_for_question",
        route_after_question,
        {
            "qa_answer": "qa_answer",
            "end": END,
        },
    )

    # Loop: after answering, go back to asking a new question
    graph.add_edge("qa_answer", "ask_for_question")

    return graph.compile()
