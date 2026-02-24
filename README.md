# TravelPlanner Benchmark: Multi-Constraint Travel Planning

Evaluates **OpenSymbolicAI** on the [TravelPlanner](https://osu-nlp-group.github.io/TravelPlanner/) benchmark (ICML 2024) — a challenging test of real-world planning where agents must produce complete multi-day travel itineraries satisfying budget, transportation, cuisine, and accommodation constraints.

Even GPT-4 achieves only a **0.6% final pass rate** on this benchmark. The task requires gathering information from multiple sources, tracking costs, respecting constraints, and assembling a coherent day-by-day plan — exactly the kind of structured, multi-step reasoning that the GoalSeeking pattern is designed for.

## Results

**OpenSymbolicAI achieves 100% on train, 99.4% on validation, and 97.9% on the full 1,000-task test set** — near-perfect scores on every commonsense and hard constraint check, with zero errors and 100% delivery rate.

### Framework Comparison at a Glance

Same model, same tools, same evaluation — only the framework differs. Full analysis in **[COMPARISON.md](COMPARISON.md)**. For multi-model results across 11 LLMs and 4 providers, see **[MODEL-LANDSCAPE.md](MODEL-LANDSCAPE.md)**.

| | OpenSymbolicAI | LangChain | CrewAI |
|---|:-:|:-:|:-:|
| **Pass Rate** | **100%** | 77.8% | 73.3% |
| **Tokens / Task** | **13,936** | 43,801 (3.1x) | 81,331 (5.8x) |
| **LLM Calls / Task** | **2.3** | 13.5 (5.9x) | 39.6 (17x) |
| **Cost / Passing Task** | **$0.013** | $0.051 (4.1x) | $0.100 (8x) |
| **Latency** | **47s** | 73s (1.5x) | 124s (2.6x) |

> Multipliers show how much more each framework consumes relative to OpenSymbolicAI. Measured on 45 train tasks (15 easy + 15 medium + 15 hard) with `gpt-oss-120b` via Fireworks AI.

### Train Split (45 tasks — full split)

| Level | Tasks | Delivery | Commonsense | Hard Constraints | Final Pass | Avg Time |
|-------|------:|--------:|:-----------:|:----------------:|:----------:|--------:|
| Easy | 15 | 100% | 100% | 100% | **100%** | — |
| Medium | 15 | 100% | 100% | 100% | **100%** | — |
| Hard | 15 | 100% | 100% | 100% | **100%** | — |
| **All** | **45** | **100%** | **100%** | **100%** | **100%** | **52.6s** |

### Validation Split (180 tasks — full split)

| Level | Tasks | Delivery | Commonsense | Hard Constraints | Final Pass | Avg Time |
|-------|------:|--------:|:-----------:|:----------------:|:----------:|--------:|
| Easy | 60 | 100% | 100% | 100% | **100%** | — |
| Medium | 60 | 100% | 98.3% | 100% | **98.3%** | — |
| Hard | 60 | 100% | 100% | 100% | **100%** | — |
| **All** | **180** | **100%** | **99.4%** | **100%** | **99.4%** | **55.5s** |

### Test Split (1,000 tasks — full split)

| Level | Tasks | Delivery | Commonsense | Hard Constraints | Final Pass | Avg Time |
|-------|------:|--------:|:-----------:|:----------------:|:----------:|--------:|
| Easy | 348 | 100% | 98.6% | 100% | **98.6%** | — |
| Medium | 333 | 100% | 97.0% | 100% | **97.0%** | — |
| Hard | 319 | 100% | 98.1% | 100% | **98.1%** | — |
| **All** | **1,000** | **100%** | **97.9%** | **100%** | **97.9%** | **52.4s** |

All hard constraint checks pass at 100% across all splits. Commonsense micro averages 99.7% on the full test set:

| Constraint Category | Check | Train | Validation | Test (1,000) |
|---|---|:-:|:-:|:-:|
| Commonsense | Within Sandbox (entities exist in DB) | 100% | 100% | 100% |
| Commonsense | Complete Information (no missing fields) | 100% | 100% | 99.9% |
| Commonsense | Within Current City (activities match city) | 100% | 100% | 99.6% |
| Commonsense | Reasonable City Route (origin → dest → origin) | 100% | 100% | 99.5% |
| Commonsense | Diverse Restaurants (no duplicates) | 100% | 100% | 98.9% |
| Commonsense | Diverse Attractions (no duplicates) | 100% | 100% | 100% |
| Commonsense | Non-Conflicting Transport (single mode) | 100% | 100% | 100% |
| Commonsense | Valid Accommodation (minimum nights) | 100% | 100% | 100% |
| Hard | Budget (total cost within limit) | 100% | 100% | 100% |
| Hard | Room Rule (house rules compliance) | 100% | 100% | 100% |
| Hard | Room Type (entire home/private/shared) | 100% | 100% | 100% |
| Hard | Cuisine (all required cuisines covered) | 100% | 100% | 100% |
| Hard | Transportation (forbidden mode not used) | 100% | 100% | 100% |

### Comparison with Published Baselines

Results from the [TravelPlanner paper](https://arxiv.org/abs/2402.01622) (ICML 2024) on the validation split:

| Method | Delivery | Commonsense | Hard | Final Pass |
|--------|:--------:|:-----------:|:----:|:----------:|
| GPT-3.5-Turbo | 100% | 2.9% | 1.7% | 0.6% |
| GPT-4 | 100% | 6.4% | 3.7% | 0.6% |
| GPT-4-Turbo | 99.4% | 11.7% | 4.6% | 4.4% |
| Gemini 1.5 Pro | 98.3% | 7.8% | 4.5% | 3.9% |
| **OpenSymbolicAI (ours)** | **100%** | **99.4%** | **100%** | **99.4%** |

> Model: `gpt-oss-120b` via Fireworks AI. Each task uses 1 retrieval iteration + 1 assembly iteration (2 LLM calls total). No retries needed. All splits are full: train (45), validation (180), test (1,000). The 0.6% validation miss is a single commonsense constraint (within_current_city) on one medium task. Hard constraints are 100% across all 1,225 tasks.

## What is TravelPlanner?

TravelPlanner gives the agent a natural language travel request and asks it to produce a complete itinerary:

> **Query:** Plan a 3-day trip from Sarasota to Chicago for 1 person with a budget of $1,900, from March 22nd to March 24th, 2022.

The agent must:
1. **Search** for flights, restaurants, accommodations, and attractions
2. **Plan** a day-by-day itinerary with transportation, meals, sightseeing, and lodging
3. **Satisfy** all explicit constraints (budget, cuisine, room type, etc.)
4. **Respect** commonsense rules (no duplicate restaurants, valid city routes, etc.)

### Difficulty Levels

| Level | Description | Constraints |
|-------|-------------|-------------|
| **Easy** | Single city, 1 person | Budget only |
| **Medium** | Single/multi city, 2-8 people | Budget + 1 constraint (cuisine, room type, or room rule) |
| **Hard** | Multi-city, variable group | Budget + 3 constraints (cuisine + room type/rule + transportation) |

### Dataset

| Split | Size | Purpose |
|-------|------|---------|
| Train | 45 | Human-annotated reference plans |
| Validation | 180 | Evaluation with ground truth |
| Test | 1,000 | Blind test set |

Source: [HuggingFace `osunlp/TravelPlanner`](https://huggingface.co/datasets/osunlp/TravelPlanner)

## Evaluation Metrics

The benchmark evaluates four categories with 13 individual checks:

### Commonsense Constraints (8 checks)

| Check | Description |
|-------|-------------|
| Within Sandbox | All entities (flights, restaurants, hotels, attractions) exist in the database |
| Complete Information | No excessive missing fields in the itinerary |
| Within Current City | Daily activities match the designated city |
| Reasonable City Route | Starts from origin, returns to origin, logical sequence |
| Diverse Restaurants | No restaurant visited more than once |
| Diverse Attractions | No attraction visited more than once |
| Non-Conflicting Transport | No mixing self-driving with flights |
| Valid Accommodation | Meets minimum-nights requirements |

### Hard Constraints (5 checks)

| Check | Description |
|-------|-------------|
| Budget | Total cost (flights + meals + accommodation) within stated budget |
| Room Rule | Accommodation complies with house rules (no smoking, no parties, etc.) |
| Room Type | Correct room type (entire home, private room, shared room) |
| Cuisine | All required cuisines represented in meals |
| Transportation | Forbidden transport mode not used (e.g., "no flights") |

### Aggregate Scores

- **Delivery Rate** — Did the agent produce a parseable plan?
- **Commonsense Macro** — Fraction of plans passing ALL 8 commonsense checks
- **Hard Macro** — Fraction of plans passing ALL applicable hard checks
- **Final Pass Rate** — Plans passing both commonsense AND hard constraints (headline metric)

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **opensymbolicai-core** — the core GoalSeeking framework
- An API key for at least one LLM provider

## Installation

```bash
git clone https://github.com/OpenSymbolicAI/benchmark-py-TravelPlanner.git
cd benchmark-py-TravelPlanner
uv sync
```

## Configuration

Create a `.env` file in the project root:

```env
# LLM provider (pick one based on --provider flag)
FIREWORKS_API_KEY=...           # Default provider - https://fireworks.ai
OPENAI_API_KEY=sk-...           # For --provider openai
ANTHROPIC_API_KEY=sk-ant-...    # For --provider anthropic
GROQ_API_KEY=gsk_...            # For --provider groq
```

## Usage

### Quick Start

```bash
# Run 5 easy validation tasks with GPT-4o
uv run travelplanner-bench --model gpt-4o --provider openai --level easy -n 5

# Run 5 easy tasks with Fireworks
uv run travelplanner-bench --model gpt-oss-120b --provider fireworks --level easy -n 5
```

### Run by Difficulty

```bash
# Easy tasks only (budget constraint, single city)
uv run travelplanner-bench --model gpt-4o --provider openai --level easy

# Medium tasks (budget + 1 constraint)
uv run travelplanner-bench --model gpt-4o --provider openai --level medium

# Hard tasks (budget + 3 constraints, multi-city)
uv run travelplanner-bench --model gpt-4o --provider openai --level hard
```

### Full Validation Set (180 tasks)

```bash
uv run travelplanner-bench --model gpt-4o --provider openai --parallel 5
```

### More Models

```bash
# Anthropic Claude
uv run travelplanner-bench --model claude-sonnet-4-20250514 --provider anthropic -n 10

# Fireworks (open-source models)
uv run travelplanner-bench --model gpt-oss-120b --provider fireworks -n 10

# Local Ollama
uv run travelplanner-bench --model llama3 --provider ollama -n 5
```

### Observability

Send structured traces to the local observability stack for per-span inspection of planning, execution, and goal-seeking iterations:

```bash
# Start the observability stack (collector + dashboard)
cd /path/to/OpenSymbolicAI/observability && docker compose up

# Run with tracing enabled
uv run travelplanner-bench --model gpt-oss-120b --provider fireworks -n 5 --observe
```

Traces are sent to `http://localhost:8100/events` and viewable in the dashboard at `http://localhost:8101`. Each task emits traces for all three agent layers:

- **TravelPlannerAgent** — top-level GoalSeeking orchestrator spans
- **RetrievalAgent** — GoalSeeking iteration spans (search primitives, evaluations)
- **PlanAssemblerAgent** — DesignExecute spans (plan generation, execution steps)

### CLI Reference

```
uv run travelplanner-bench --model MODEL --provider PROVIDER [OPTIONS]

Required:
  --model MODEL                                       Model name/ID (e.g., gpt-4o, gpt-oss-120b)
  --provider {ollama,openai,anthropic,fireworks,groq}  LLM provider

Options:
  --split {train,validation,test}                      Dataset split (default: validation)
  -l, --level {easy,medium,hard}                       Filter by difficulty level
  -n, --num NUM                                        Number of tasks (default: all)
  --max-iterations N                                   Max agent iterations per task (default: 10)
  -p, --parallel N                                     Parallel workers (default: 3)
  --shuffle                                            Shuffle tasks
  --seed SEED                                          Random seed (default: 42)
  --observe                                            Enable observability traces (http://localhost:8100)
```

## Output

Each run creates a timestamped directory under `logs/`:

```
logs/<timestamp>_<model>/
  summary.json          # Aggregate metrics (delivery rate, constraint scores, final pass rate)
  results.json          # Per-task results
  task_0001_tp_0000.md  # Detailed per-task log with plan and constraint results
  agent_debug.log       # Full agent iteration trace
```

### summary.json

Contains delivery rate, commonsense/hard constraint micro/macro scores, final pass rate, per-level breakdown, and timing.

### task_NNNN.md

Each file contains:
- Original query and constraints
- Plan delivered (yes/no) and final pass (yes/no)
- All 8 commonsense + 5 hard constraint results
- The generated JSON itinerary
- Error details if the agent failed

## Architecture

### GoalSeeking Two-Stage Pattern

```
Travel Query + Constraints
        |
        v
  [Stage 1: Information Gathering]
        |
        v
  Plan LLM Call --> Python code using search primitives
        |
        v
  Execute: search_flights, search_restaurants,
           search_accommodations, search_attractions,
           get_distance, search_cities
        |
        v
  Introspect into TravelPlanContext
  (flights_found, restaurants_found, ...)
        |
        v
  Evaluate: enough data gathered?
        |           |
       No           Yes
        |             |
  next iteration      v
                [Stage 2: Plan Assembly]
                      |
                      v
                Plan LLM Call --> Python code building itinerary
                      |
                      v
                Execute: set_plan([day1, day2, ...])
                      |
                      v
                Evaluate: plan submitted? --> Done
```

### Key Design Decisions

- **1 LLM call per iteration** generates Python code with multiple primitive calls
- **ReferenceDatabase** indexes each task's pre-collected data (flights, restaurants, etc.) for tool queries
- **Symbolic firewall**: raw search results stay in app memory; the LLM sees structured context
- **set_plan()** as terminal primitive: the agent explicitly decides when the plan is complete
- **Evaluation against same database**: constraint checks validate against the same reference data the tools provide

### Agent Primitives

| Primitive | Stage | Description |
|-----------|-------|-------------|
| `search_flights(origin, dest, date)` | Gather | Find flights between cities on a date |
| `search_restaurants(city)` | Gather | Find restaurants in a city |
| `search_accommodations(city)` | Gather | Find hotels/apartments in a city |
| `search_attractions(city)` | Gather | Find attractions in a city |
| `get_distance(origin, dest, mode)` | Gather | Get driving/taxi distance and cost |
| `search_cities(state)` | Gather | List cities in a state (for multi-city trips) |
| `set_plan(plan)` | Build | Submit the final day-by-day itinerary |

## Framework Comparison Runner

Compare OpenSymbolicAI against LangChain and CrewAI on the same tasks, same model, side by side. Measures **token utilization** and **reliability** across frameworks.

### Install Comparison Dependencies

```bash
# LangChain only
uv add --optional langchain "langchain-core>=0.3.0" "langchain-openai>=0.3.0" "langgraph>=0.2.0"

# CrewAI only
uv add --optional crewai "crewai>=0.80.0"

# Both (for full comparison)
uv sync --extra langchain --extra crewai
```

### Run a Comparison

```bash
# All 3 frameworks on the train split (45 tasks)
uv run travelplanner-compare \
    --frameworks opensymbolicai,langchain,crewai \
    --model gpt-4o --provider openai

# Quick 5-task test with LangChain
uv run travelplanner-compare \
    --frameworks langchain --model gpt-4o --provider openai --num 5

# Two-framework head-to-head on easy tasks
uv run travelplanner-compare \
    --frameworks opensymbolicai,langchain --level easy \
    --model gpt-4o --provider openai

# Specific tasks only
uv run travelplanner-compare \
    --frameworks opensymbolicai,crewai \
    --model gpt-4o --provider openai \
    --task-ids tp_0003,tp_0010,tp_0015
```

### Comparison CLI Reference

```
uv run travelplanner-compare [OPTIONS]

Required:
  --model MODEL                                       LLM model (same for all frameworks)
  --provider {openai,anthropic,fireworks,groq,ollama}  LLM provider

Options:
  -f, --frameworks LIST                  Comma-separated frameworks (default: all three)
  --split {train,validation,test}        Dataset split (default: train)
  -l, --level {easy,medium,hard}         Filter by difficulty
  -n, --num NUM                          Number of tasks (default: all)
  --max-iterations N                     Max agent iterations per task (default: 10)
  -p, --parallel N                       Parallel workers per framework (default: 1)
  --task-ids IDS                         Comma-separated task IDs to run
```

### Comparison Output

Each run creates a directory under `logs/`:

```
logs/compare_<timestamp>/
  comparison_report.md      # Side-by-side Markdown tables
  comparison_summary.json   # Machine-readable metrics
  opensymbolicai/
    results.json            # Per-task results
    task_0001_tp_0000.md    # Detailed per-task logs
  langchain/
    results.json
    task_0001_tp_0000.md
  crewai/
    results.json
    task_0001_tp_0000.md
```

### Metrics Compared

**Reliability** — delivery rate, final pass rate, commonsense/hard constraint macro rates, error rate

**Token Efficiency** — total tokens, avg tokens/task, retrieval vs assembly split, LLM calls/task, estimated cost (USD), cost per passing task

**Timing** — avg/p50/p95 wall time per task

### How Each Backend Works

| Framework | Retrieval Phase | Assembly Phase | Post-processing |
|-----------|----------------|----------------|-----------------|
| **OpenSymbolicAI** | GoalSeeking agent iteratively calls search primitives via LLM-generated Python code | DesignExecute agent generates plan via LLM-generated Python code | Shared 8-phase `_fill_missing_fields` |
| **LangChain** | `create_react_agent` (langgraph) with 6 search tools | Single structured LLM call to generate plan JSON | Same shared post-processing |
| **CrewAI** | Sequential Crew: Researcher agent with search tools | Planner agent receives research context, outputs plan JSON | Same shared post-processing |

All three frameworks share the same `ReferenceDatabase`, search primitives, evaluation pipeline, and deterministic post-processing. This isolates the comparison to framework overhead and LLM interaction patterns.

## Project Structure

```
benchmark-py-TravelPlanner/
  travelplanner_bench/
    __init__.py              # Package exports
    models.py                # TravelPlannerTask, TravelPlanContext, TravelPlannerResult
    data.py                  # HuggingFace dataset loader + JSON parsing
    tools.py                 # ReferenceDatabase + 6 search tool functions
    agent.py                 # TravelPlannerAgent (GoalSeeking)
    evaluation.py            # 8 commonsense + 5 hard constraint checks
    runner.py                # CLI benchmark runner (single framework)
    backend.py               # AgentBackend protocol, TokenUsage, BackendResult
    token_tracking.py        # Model pricing + token extraction helpers
    tool_wrappers.py         # LangChain/CrewAI tool adapters
    comparison_runner.py     # Multi-framework comparison CLI
    comparison_report.py     # Side-by-side Markdown + JSON report generator
    backends/
      __init__.py            # Backend registry
      opensymbolicai_backend.py  # Wraps existing TravelPlannerAgent
      langchain_backend.py   # LangChain ReAct agent
      crewai_backend.py      # CrewAI Crew with 2 agents
  tests/
    test_models.py           # Model creation and serialization tests
    test_data.py             # Data parsing tests
    test_tools.py            # ReferenceDatabase and search function tests
    test_evaluation.py       # All 13 constraint checker tests
  logs/                      # Per-run logs
  main.py                    # Entry point
  pyproject.toml
```

## Tests

```bash
# Run unit tests
uv run pytest

# With coverage
uv run pytest --cov=travelplanner_bench

# Verbose output
uv run pytest -v
```

## References

- [TravelPlanner: A Benchmark for Real-World Planning with Language Agents](https://arxiv.org/abs/2402.01622) (ICML 2024)
- [HuggingFace Dataset: osunlp/TravelPlanner](https://huggingface.co/datasets/osunlp/TravelPlanner)
- [TravelPlanner Project Page](https://osu-nlp-group.github.io/TravelPlanner/)
- [TravelPlanner GitHub](https://github.com/OSU-NLP-Group/TravelPlanner)
