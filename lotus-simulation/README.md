# lotus-simulation

LOTUS-based simulation scripts for three agent-memory insertion pipelines:
- `mem0_lotus.py`
- `evermemos_lotus.py`
- `zep_lotus.py`

All runtime settings are in one YAML file (`config.yaml` by default).
Secrets stay in environment variables.

## Environment Setup

From `/Users/von/Projects/AgentMemAnalysis/lotus-simulation`:

```bash
# venv + pip
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ../lotus
pip install python-dotenv pyyaml
```

```bash
# uv
uv venv --python 3.11
source .venv/bin/activate
uv add --editable ../lotus
uv add python-dotenv pyyaml
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
- `models.rm`: retrieval backend (`hashing` / `sentence_transformers` / `litellm`)
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
    backend: hashing
    model: intfloat/e5-base-v2
    kwargs:
      dim: 1024

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

### Retrieval Backends (`models.rm`)

- `hashing` (default): no torch, no external embedding API, easiest to run everywhere.
- `sentence_transformers`: local embedding model (needs `torch` + `sentence-transformers`).
- `litellm`: embedding through LiteLLM provider API (no local torch needed).

Examples:

```yaml
# local sentence-transformers
models:
  rm:
    backend: sentence_transformers
    model: intfloat/e5-base-v2
    kwargs: {}
```

```yaml
# API embedding via LiteLLM
models:
  rm:
    backend: litellm
    model: text-embedding-3-small
    kwargs: {}
```

## Torch Import Hang Fix

If running `python evermemos_lotus.py` hangs during `import torch`:

1. Use non-torch RM backend:

```yaml
models:
  rm:
    backend: hashing
    model: intfloat/e5-base-v2
    kwargs:
      dim: 1024
```

2. Re-run:

```bash
python evermemos_lotus.py
```

3. Optional diagnostics:

```bash
python -c "import lotus,sys; print('torch_loaded', 'torch' in sys.modules)"
python -c "import torch; print(torch.__version__)"
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

## vLLM on Server (Llama-3.1-70B)

Start OpenAI-compatible vLLM endpoint:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --port 8001
```

Then set YAML:

```yaml
models:
  helper:
    enabled: true
    model: hosted_vllm/meta-llama/Llama-3.1-70B-Instruct
    max_tokens: 512
    temperature: 0.0
    max_batch_size: 8
    kwargs:
      api_base: http://127.0.0.1:8001/v1
      api_key: EMPTY
```

Note: bf16 70B typically needs >1 GPU (or quantization). If you have a single H100 80GB, use a smaller model (e.g. 32B/7B) or quantized 70B.

## Optimization Switches (On/Off)

Edit `/Users/von/Projects/AgentMemAnalysis/lotus-simulation/config.yaml`:

- enable all:

```yaml
optimizations:
  filter_cascade_enabled: true
  join_cascade_enabled: true
  topk_cascade_enabled: true
  proxy_model: helper_lm
  recall_target: 0.9
  precision_target: 0.9
  failure_probability: 0.2
  sampling_percentage: 0.2
  min_join_cascade_size: 100
```

- disable all:

```yaml
optimizations:
  filter_cascade_enabled: false
  join_cascade_enabled: false
  topk_cascade_enabled: false
  proxy_model: embedding
```

Notes:
- LOTUS `sem_filter` cascade supports both embedding proxy and helper-LM proxy.
- LOTUS `sem_join` cascade is primarily similarity-proxy driven in current code path; helper LM is not the main join proxy path.
- `sim_config.py` is now the single config loader (including optimization config); `lotus_opt_config.py` has been merged away.
