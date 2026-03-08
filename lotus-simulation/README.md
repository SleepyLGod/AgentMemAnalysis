# lotus-simulation

LOTUS-based simulation scripts for three agent-memory insertion pipelines:
- `mem0_lotus.py`
- `evermemos_lotus.py`
- `zep_lotus.py`

Each script runs on LOCOMO conversation data, calls an LLM via LiteLLM, and writes a full JSON execution log.

## 1) Environment Setup

From `lotus-simulation`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ../lotus
pip install python-dotenv
```

Set API key in repo root `.env` (or shell env):

```bash
DEEPSEEK_API_KEY=your_key_here
```

Notes:
- Python: `>=3.10,<3.13` (aligned with LOTUS).
- All scripts use `lotus.settings.configure(..., enable_cache=True)`.
- Default LOTUS cache in this setup is exact-input hash cache (in-memory per process).
- All optimization toggles are OFF by default, so baseline behavior is unchanged.

Optional optimization env vars (all default OFF / baseline-equivalent):

```bash
LOTUS_OPT_FILTER_CASCADE=0
LOTUS_OPT_JOIN_CASCADE=0
LOTUS_OPT_TOPK_CASCADE=0
LOTUS_OPT_PROXY_MODEL=embedding      # embedding | helper_lm
LOTUS_OPT_RECALL_TARGET=0.9
LOTUS_OPT_PRECISION_TARGET=0.9
LOTUS_OPT_FAILURE_PROB=0.2
LOTUS_OPT_SAMPLING_PCT=0.2
LOTUS_OPT_MIN_JOIN_CASCADE_SIZE=100
LOTUS_HELPER_MODEL=                  # required only when LOTUS_OPT_PROXY_MODEL=helper_lm
```

## 2) Run

```bash
cd /Users/von/Projects/AgentMemAnalysis/lotus-simulation
source .venv/bin/activate

python mem0_lotus.py
python evermemos_lotus.py
python zep_lotus.py
```

### Run with env vars (copy/paste)

Baseline (all optimizations OFF, behavior-equivalent to original flow):

```bash
cd /Users/von/Projects/AgentMemAnalysis/lotus-simulation
source .venv/bin/activate

export LOTUS_OPT_FILTER_CASCADE=0
export LOTUS_OPT_JOIN_CASCADE=0
export LOTUS_OPT_TOPK_CASCADE=0

python mem0_lotus.py
python evermemos_lotus.py
python zep_lotus.py
```

Join cascade only (mainly affects `zep_lotus.py`):

```bash
cd /Users/von/Projects/AgentMemAnalysis/lotus-simulation
source .venv/bin/activate

export LOTUS_OPT_JOIN_CASCADE=1
export LOTUS_OPT_FILTER_CASCADE=0
export LOTUS_OPT_PROXY_MODEL=embedding
export LOTUS_OPT_RECALL_TARGET=0.9
export LOTUS_OPT_PRECISION_TARGET=0.9
export LOTUS_OPT_FAILURE_PROB=0.2
export LOTUS_OPT_SAMPLING_PCT=0.2
export LOTUS_OPT_MIN_JOIN_CASCADE_SIZE=100

python zep_lotus.py
```

Filter cascade only (affects `mem0_lotus.py` Q8 and `evermemos_lotus.py` Q1):

```bash
cd /Users/von/Projects/AgentMemAnalysis/lotus-simulation
source .venv/bin/activate

export LOTUS_OPT_FILTER_CASCADE=1
export LOTUS_OPT_JOIN_CASCADE=0
export LOTUS_OPT_PROXY_MODEL=embedding
export LOTUS_OPT_RECALL_TARGET=0.9
export LOTUS_OPT_PRECISION_TARGET=0.9
export LOTUS_OPT_FAILURE_PROB=0.2
export LOTUS_OPT_SAMPLING_PCT=0.2

python mem0_lotus.py
python evermemos_lotus.py
```

Helper-LM proxy for cascade (optional):

```bash
cd /Users/von/Projects/AgentMemAnalysis/lotus-simulation
source .venv/bin/activate

export LOTUS_OPT_FILTER_CASCADE=1
export LOTUS_OPT_PROXY_MODEL=helper_lm
export LOTUS_HELPER_MODEL=deepseek/deepseek-chat

python evermemos_lotus.py
```

Generated logs:
- `mem0_execution_log_YYYYMMDD_HHMMSS.json`
- `lotus_execution_log_YYYYMMDD_HHMMSS.json` (EverMemOS simulation)
- `zep_execution_log_YYYYMMDD_HHMMSS.json`

## 3) Script Overview

### `mem0_lotus.py`
- Simulates Mem0 insertion workflow with LOTUS semantic operators.
- Core stages: fact/entity/relation extraction, similarity recall, conflict/no-op/new-memory decisions.
- Creates local FAISS index folders: `mem0_fact_idx/`, `mem0_ent_idx/`, `mem0_rel_idx/`.
- Default scope: first `40` messages.

### `evermemos_lotus.py`
- Simulates EverMemOS segment/topic/profile insertion workflow.
- Core stages: boundary detection, segment extraction, topic assignment/creation, profile distillation.
- Includes optional time-gap gating when session timestamps are parseable.
- Default scope: first `40` messages.

### `zep_lotus.py`
- Simulates Graphiti/Zep insertion workflow.
- Core stages: node/edge extraction, entity resolution, edge dedup/contradiction checks, summary refresh.
- Includes one-time optional community bootstrap (`AUTO_BUILD_COMMUNITIES_AFTER_MSG`) before insertion-time community updates.
- Default scope: first `40` messages, sliding context window `10`.
