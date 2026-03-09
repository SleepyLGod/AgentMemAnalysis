# lotus-simulation

LOTUS-based simulation scripts for three agent-memory insertion pipelines:
- `mem0_lotus.py`
- `evermemos_lotus.py`
- `zep_lotus.py`

All non-secret runtime settings are loaded from YAML (`config.yaml` by default).
Secrets stay in environment variables.

## Environment Setup

From `/Users/von/Projects/AgentMemAnalysis/lotus-simulation`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ../lotus
pip install python-dotenv pyyaml
```

Set API key in shell or repo `.env`:

```bash
DEEPSEEK_API_KEY=your_key_here
```

## Run

```bash
cd /Users/von/Projects/AgentMemAnalysis/lotus-simulation
source .venv/bin/activate

python mem0_lotus.py
python evermemos_lotus.py
python zep_lotus.py
```

Use a custom config file:

```bash
LOTUS_SIM_CONFIG_PATH=/abs/path/config.yaml python mem0_lotus.py
```

Generated logs:
- `mem0_execution_log_YYYYMMDD_HHMMSS.json`
- `lotus_execution_log_YYYYMMDD_HHMMSS.json` (EverMemOS simulation)
- `zep_execution_log_YYYYMMDD_HHMMSS.json`

## Configuration

Default file: `/Users/von/Projects/AgentMemAnalysis/lotus-simulation/config.yaml`

Top-level schema:
- `global`: `locomo_path`, `conv_index`, `enable_cache`
- `models.main`: main LM config
- `models.helper`: helper LM config
- `models.rm`: retrieval model (embedding)
- `optimizations`: cascade/proxy settings
- `pipelines.mem0`: Mem0-specific settings
- `pipelines.zep`: Zep-specific settings
- `pipelines.evermemos`: EverMemOS-specific settings

Minimal template:

```yaml
global:
  locomo_path: ../evermemos/evaluation/data/locomo/locomo10.json
  conv_index: 0
  enable_cache: true

models:
  main:
    model: deepseek/deepseek-chat
    max_tokens: 1000
    temperature: 0.0
    max_batch_size: 2
    kwargs: {}
  helper:
    enabled: false
    model: ""
    max_tokens: 1000
    temperature: 0.0
    max_batch_size: 2
    kwargs: {}
  rm:
    model: intfloat/e5-base-v2

optimizations:
  filter_cascade_enabled: false
  join_cascade_enabled: false
  topk_cascade_enabled: false
  proxy_model: embedding
  recall_target: 0.9
  precision_target: 0.9
  failure_probability: 0.2
  sampling_percentage: 0.2
  min_join_cascade_size: 100

pipelines:
  mem0:
    max_messages: 40
    sim_noop_threshold: 0.92
  zep:
    max_messages: 40
    sliding_window_size: 10
    auto_build_communities_after_msg: 6
  evermemos:
    max_messages: 40
    boundary_msg_threshold: 15
    topic_sim_threshold: 0.5
    topic_max_gap_days: 7
    profile_min_segments: 2
```

## Helper LM Examples

Helper LM is configured in YAML and only used when:
- `optimizations.proxy_model: helper_lm`
- `models.helper.enabled: true`
- `models.helper.model` is non-empty

Example 1 (hosted API helper):

```yaml
models:
  helper:
    enabled: true
    model: deepseek/deepseek-chat
    max_tokens: 1000
    temperature: 0.0
    max_batch_size: 2
    kwargs: {}
```

Example 2 (local vLLM helper, e.g. Qwen/Llama):

```yaml
models:
  helper:
    enabled: true
    model: hosted_vllm/Qwen/Qwen2.5-7B-Instruct
    max_tokens: 512
    temperature: 0.0
    max_batch_size: 8
    kwargs:
      api_base: http://127.0.0.1:8000/v1
      api_key: EMPTY
```

Notes:
- LOTUS `sem_filter` cascade supports both embedding proxy and helper-LM proxy.
- LOTUS `sem_join` cascade is primarily similarity-proxy driven in current code path; helper LM is not the main join proxy path.
- All optimization switches default to OFF in `config.yaml`, so baseline behavior remains unchanged.
