# Clash Royale Data Pipeline & Coaching Agent
## Author
Built by **Hien Tran**

Computer Science @ California State University, Fullerton  

[LinkedIn](link) • [GitHub](link)

**Video Demo:** [Watch on LinkedIn](link)

A three-phase analytics & coaching system for **Clash Royale** built with:

- **Python + pandas + Matplotlib** for data processing & plots  
- **Clash Royale Developer API** for battle logs  
- **LangGraph + OpenAI** for orchestration and LLM-powered coaching  
- **LangSmith Studio** for interactive graph visualization and demo

The agent:

1. Builds a **global meta dataset** from top-ladder players.  
2. Computes **per-player analytics** for a given tag.  
3. Uses an LLM-based **Q&A coach** to answer questions using both meta + personal stats.

---
# Table of Contents
- [Overview -- High-Level Architecture](#high-level-architecture)
- [Phase 0 —- Meta Analytics](#phase-0--meta-analytics)
- [Phase 1 —- User Analytics](#phase-1--user-analytics)
- [Phase 2 —- Q&A Coaching Workflow](#phase-2--qa-coaching-workflow)
- [Combined Coach Workflow](#combined-coach-workflow)
- [Classification](#data-cleaning--deck-classification)
- [Quickstart Guide](#quickstart-guide)

---

## Diagrams

- Phase 0 – Meta pipeline in LangGraph Studio  
  ![Phase 0 – Meta Graph](docs/images/phase0_meta_graph.png)

- Phase 1 – User analytics pipeline  
  ![Phase 1 – User Graph](docs/images/phase1_user_graph.png)

- Phase 2 – Q&A coach graph  
  ![Phase 2 – QnA Graph](docs/images/phase2_qna_graph.png)

- Combined “Coach” graph (Phase 0 + 1 + 2)  
  ![Combined Coach Graph](docs/images/coach_graph.png)

---

## High-Level Architecture

The project is organized into **three phases** plus a **top-level coach**:

1. **Phase 0 – Meta Pipeline**  
   Build a reference dataset from top players:
   - Sample high-ladder players via Clash Royale API.
   - Fetch their recent ranked 1v1 games.
   - Normalize battles and compute **global meta stats**.
   - Keep sampling until there are enough total games and for each major archetype.
   - Build a **LLM-friendly table** and plots.

2. **Phase 1 – User Analytics**  
   For a single player tag:
   - Fetch their battle log.
   - Filter to **ranked / Trophy Road 1v1** only.
   - Compute summary stats, deck-type performance, card stats, and matchups.
   - Generate plots.
   - Build **LLM-friendly table** summarizing the user.

3. **Phase 2 – Q&A Coach**  
   - Takes a natural-language question.
   - Uses a cheap LLM to classify the question and decide what data is needed.
   - Prepares the right context slices from **user + meta** tables.
   - Calls an expert model to produce a **coaching answer**.

4. **Coach Graph**  
   - Ensures Phase 0 meta data exists (once per session).  
   - Interrupts to ask for a **player tag**.  
   - Runs Phase 1 analytics for that tag.  
   - Interrupts to ask for a **question**.  
   - Calls Phase 2 to answer.  
   - Loops back to “ask for question” so the user can keep chatting with the coach.

All of these are implemented as **LangGraph graphs**, visualized and run via **LangSmith Studio**.

---

## Phase 0 – Meta Dataset Builder (`src/workflows/meta_workflow.py`)

**Goal:** Build a high-quality, balanced meta dataset from top-ladder players.

## Graphs
![Phase 0 – Meta Graph](docs/images/phase0_meta_graph.png)
### Node

1. **Fetch top players**  
   - `fetch_top_players_node` calls the Clash Royale API (the top 1000 players).
   - State keeps `top_players`, `used_player_indices`, and `fetched_player_tags`.

2. **Initial sampling**  
   - `sample_initial_node` randomly selects an initial cohort (e.g. 250 players).

3. **Fetch & normalize battles**  
   - `fetch_meta_battles_node`:
     - Fetches each selected player’s battle log.
     - Filters / normalizes to ranked/Trophy Road only using `filter_and_normalize_ranked_1v1`.
     - Keeps up to **10 most recent** ranked 1v1 games per player.
     - Appends to `meta_raw_battles`.

4. **Compute meta analytics**  
   - `compute_meta_analytics_node` calls `compute_meta_analytics` to compute:
     - Global summary (`games_total`, overall win rate, etc.).
     - Deck-type counts (from both ranker & ranker opponent).
     - Deck-type vs deck-type matchup matrix.

5. **Stopping condition**  
   - `check_enough_battles_node` enforces:
     - `MIN_TOTAL_BATTLES` (e.g. 2000 games).  
     - For each required deck type, games ≥ `MIN_GAMES_PER_TYPE`.  
       - Required types: **Siege, Bait, Cycle, Bridge Spam, Beatdown**  
       - Hybrid is allowed to be sparse.
   - Returns a `stop_decision`:
     - `"enough"` → move on to finalization.  
     - `"need_more"` → sample 5 more unused players via `sample_more_5_node`.  
     - `"stop"` → no players / max loops → finalize with what we have.

6. **Standardize meta table**  
   - `standardize_meta_table_node` builds a **row-per-participant** table (`meta_table`) using `build_standardized_meta_table`.  
   - Each row includes deck type, result, and other normalized features.

7. **LLM-friendly meta tables**  
   - `build_meta_llm_tables_node` consumes `meta_table` + the matchup matrix to build:
     - `meta_deck_summary` – one row per archetype with games, wins, losses, win rate, etc.  
     - `meta_matchup_summary` – one row per archetype vs archetype matchup with win rate & sample size.  
   - Stored under `state["meta_llm_tables"]`.

8. **Meta plots**  
   - `generate_meta_plots_node` uses `meta_llm_tables` to create:
     - A **pie chart** of meta deck share.  
     - A **bar chart** of deck-type win rates.  
     - One **per-deck matchup chart**: each archetype vs all others (mirror excluded), bars labeled with win rate % and total games in the title.
   - Plots are saved and stored under plots folder.

---

## Phase 1 – User Analytics Pipeline (`src/workflows/user_workflow.py`)

**Goal:** Given a **Clash Royale tag**, analyze the player’s recent performance.

## Graphs
![Phase 1 – User Analytics](docs/images/phase1_user_graph.png)

### Node

1. **Fetch battle log** – `fetch_battlelog_node`  
   - Reads `player_tag`.  
   - Calls `get_player_battlelog`.  
   - Writes `battles_raw` + note.

2. **Filter & normalize** – `filter_and_normalize_node`  
   - Uses `filter_and_normalize_ranked_1v1` to keep only ranked/Trophy Road only.  
   - Writes `battles_filtered`.

3. **Compute user analytics** – `compute_user_analytics_node`  
   - Calls `compute_user_analytics(battles_filtered)`.  
   - `user_analytics["summary"]` includes games_played, wins, losses, win rate.  
   - Additional sections: deck-type stats, matchup stats, card performance, etc.

4. **Build LLM tables** – `build_user_llm_tables_node`  
   Produces `user_llm_tables` with:

   - `user_summary` – rows `{metric, value}` from `analytics["summary"]`.  
   - `user_deck_summary` – per-deck-type stats, normalized to a `deck_type` column.  
   - `user_matchup_summary` – deck-type vs deck-type matchups (my vs opp).  
   - `user_card_summary` – merges best & worst cards, with `role = "best" | "worst"`.  
   - `opponent_card_summary` – merges tough & easy opponent cards, with `role = "tough" | "easy"`.

5. **Generate plots** – `generate_user_plots_node`  
   - Uses `generate_card_plots` to build card-level plots.  
   - Writes `user_plots` and stores paths inside `user_analytics["plots"]`.

# PHASE 2 — Question-and-Answer Coaching Workflow

Phase 2 is where the system becomes an *interactive Clash Royale coach*.  
It takes the processed data from Phase 0 (meta) and Phase 1 (user) and uses a LangGraph-driven workflow to:

1. Ask the player for a question  
2. Classify the question (cheap LLM)  
3. Decide which analytics tables are needed  
4. Build a compact LLM-ready context  
5. Call an expert model to produce actionable coaching advice  

---

### Phase 2 Workflow Diagram (Q&A Graph)

![Phase 2 – QnA Pipeline](docs/images/phase2_qna_graph.png)

---

## How Phase 2 Works

### **1. Question Classification**  
A small and fast LLM model (ex: `gpt-4.1-nano`) classifies the user’s question into one of the categories:

- `user` — personal performance questions  
- `matchup` — matchup difficulty, counters, deck-type winrates  
- `meta` — global meta questions  
- `card` — best & worst cards  
- `other` — fallback  

It also determines which data the expert model needs  
(e.g., `USER_SUMMARY`, `META_DECK_MATCHUPS`, `USER_CARD_SUMMARY`, etc.).

---

### **2. Preparing the Relevant Context**  
Based on classification, the graph routes to one of:

- `prep_user_context`  
- `prep_matchup_context`  
- `prep_card_context`  
- `prep_meta_context`  
- `prep_other_context`  

Each one assembles a minimal JSON context including only the needed tables:

- User overall summary  
- User deck performance  
- Matchup table  
- Best/worst cards  
- Opponent card difficulty  
- Meta deck summary  
- Meta matchup summary  

These match the tables generated in Phase 0 and Phase 1.

---

### **3. Data Health Check**  
If the user has fewer than **20 ranked games**, Phase 2 adds a small warning.  
This does **not block** the question — it simply tells the expert LLM to be warn the user of small data size.  
Meta-only questions skip this check entirely.

---

### **4. Expert LLM Coaching (main answer)**  
A larger and slower model (ex: `gpt-4.1-mini`) receives:

- The question  
- The chosen tables  
- A compact text summary  
- Any low-data warning  

It returns a detailed answer. 

---

# COMBINED COACH WORKFLOW (Phase 0 + Phase 1 + Phase 2)

The final system merges all phases into a single LangGraph application.

Flow:

1. `ensure_meta`  
   - Builds Phase 0 meta dataset once per Studio thread  

2. `ask_for_tag`  
   - Prompts the user for a Clash Royale player tag via interrupt()  

3. `ensure_user`  
   - Runs Phase 1 user analytics for that tag  

4. `ask_for_question`  
   - Prompts the user for any question they want to ask  

5. `qa_answer`  
   - Runs the entire Phase 2 QnA subgraph  

6. Loops back to `ask_for_question`  
   - Allows unlimited questions in the same session  

---

### Combined Coach Workflow Diagram

![Combined Coach Graph](docs/images/coach_graph.png)

---

## Data Cleaning & Deck Classification

Raw Clash Royale battlelogs contain **102 cards** and thousands of possible 8-card combinations.  
To make analytics meaningful, every deck is standardized into **six archetypes**:

| Archetype | Description |
|----------|-------------|
| **Siege** | Win condition placed at long distance (X-Bow / Mortar). |
| **Bait** | Forces inefficient responses through swarms |
| **Cycle** | Low-elixir decks focused on fast rotations. |
| **Bridge Spam** | High-pressure units deployed at the bridge to punish mistakes. |
| **Beatdown** | Slow, tank-based pushes built around elixir advantages and one big push. |
| **Hybrid** | Mixed strategies that don’t fit cleanly into one category. |

This lightweight classification ensures that:

- Phase 0 (meta analysis) and Phase 1 (user analysis) **use the same archetype system**
- Matchup tables remain **consistent & interpretable**  
- LLM coaching in Phase 2 can reference decks in natural language

---
## Quickstart Guide

This project is designed to be tested efficiently in **two ways**:

1. Through **LangGraph Studio** using `langgraph dev`  
2. Through a **Jupyter / Colab notebook** `clash_agent_demo.ipynb`

---

### 1. Set up Python environment

From the project root:

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```
### 2. Configure API keys
This project includes a `.env.example` template.

To set up your environment variables, choose one of the following options:

#### Option A — Copy the example file
Rename the example file to `.env`:

```bash
cp .env.example .env
```
Then open .env and fill in your keys

#### Option B — Create a new `.env` manually
If you prefer, create a new .env file from scratch and paste:
```bash
OPENAI_API_KEY=your_openai_key
CR_API_KEY=your_clash_royale_dev_key
```

#### Note:
If you don't have a openai key eye go to https://platform.openai.com to create one.

If you don't have a clash royale dev key go to https://developer.clashroyale.com to create one.

You can also run `testapi.py` to see if your api is working
### 3. How to run it!
#### Option A - Run in LangGraph Studio
From the project root, start the dev server by opening up a terminal and paste:
```
langgraph dev
```
1. Open the Studio UI link in your browser.
2. Select qna-full from the drop-down.
3. Click the Submit/Run bnt to start!


#### Option B — Run in Jupyter / Colab
From the project root, start the dev server by opening up a terminal and paste:

```
jupyter notebook
```
1. Open `clash_agent_demo.ipynb`.
2. Run the cells in order. 

*note follow the comment instruction!