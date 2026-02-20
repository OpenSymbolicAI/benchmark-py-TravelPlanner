# TravelPlanner Benchmark: Multi-Constraint Travel Planning

Evaluates **OpenSymbolicAI** on the [TravelPlanner](https://osu-nlp-group.github.io/TravelPlanner/) benchmark (ICML 2024) — a challenging test of real-world planning where agents must produce complete multi-day travel itineraries satisfying budget, transportation, cuisine, and accommodation constraints.

Even GPT-4 achieves only a **0.6% final pass rate** on this benchmark. The task requires gathering information from multiple sources, tracking costs, respecting constraints, and assembling a coherent day-by-day plan — exactly the kind of structured, multi-step reasoning that the GoalSeeking pattern is designed for.

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
# Run 5 easy validation tasks (default: Fireworks gpt-oss-120b)
uv run travelplanner-bench --level easy -n 5

# Or via python
uv run python main.py --level easy -n 5
```

### Run by Difficulty

```bash
# Easy tasks only (budget constraint, single city)
uv run travelplanner-bench --level easy

# Medium tasks (budget + 1 constraint)
uv run travelplanner-bench --level medium

# Hard tasks (budget + 3 constraints, multi-city)
uv run travelplanner-bench --level hard
```

### Full Validation Set (180 tasks)

```bash
uv run travelplanner-bench --parallel 5
```

### Use a Different Model

```bash
# OpenAI GPT-4o
uv run travelplanner-bench -n 10 --model gpt-4o --provider openai

# Anthropic Claude
uv run travelplanner-bench -n 10 --model claude-sonnet-4-20250514 --provider anthropic

# Local Ollama
uv run travelplanner-bench -n 5 --model llama3 --provider ollama
```

### CLI Reference

```
uv run travelplanner-bench [OPTIONS]

Options:
  --model MODEL                                       Model name/ID (default: gpt-oss-120b)
  --provider {ollama,openai,anthropic,fireworks,groq}  LLM provider (default: fireworks)
  --split {train,validation,test}                      Dataset split (default: validation)
  -l, --level {easy,medium,hard}                       Filter by difficulty level
  -n, --num NUM                                        Number of tasks (default: all)
  --max-iterations N                                   Max agent iterations per task (default: 10)
  -p, --parallel N                                     Parallel workers (default: 3)
  --shuffle                                            Shuffle tasks
  --seed SEED                                          Random seed (default: 42)
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

## Project Structure

```
benchmark-py-TravelPlanner/
  travelplanner_bench/
    __init__.py          # Package exports
    models.py            # TravelPlannerTask, TravelPlanContext, TravelPlannerResult
    data.py              # HuggingFace dataset loader + JSON parsing
    tools.py             # ReferenceDatabase + 6 search tool functions
    agent.py             # TravelPlannerAgent (GoalSeeking)
    evaluation.py        # 8 commonsense + 5 hard constraint checks
    runner.py            # CLI benchmark runner
  tests/
    test_models.py       # Model creation and serialization tests
    test_data.py         # Data parsing tests
    test_tools.py        # ReferenceDatabase and search function tests
    test_evaluation.py   # All 13 constraint checker tests
  logs/                  # Per-run logs
  main.py                # Entry point
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
