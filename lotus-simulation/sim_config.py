from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SIM_CONFIG: dict[str, Any] = {
    "global": {
        "locomo_path": "../evermemos/evaluation/data/locomo/locomo10.json",
        "conv_index": 0,
        "enable_cache": True,
    },
    "models": {
        "main": {
            "model": "deepseek/deepseek-chat",
            "max_tokens": 1000,
            "temperature": 0.0,
            "max_batch_size": 2,
            "kwargs": {},
        },
        "helper": {
            "enabled": False,
            "model": "",
            "max_tokens": 1000,
            "temperature": 0.0,
            "max_batch_size": 2,
            "kwargs": {},
        },
        "rm": {"model": "intfloat/e5-base-v2"},
    },
    "optimizations": {
        "filter_cascade_enabled": False,
        "join_cascade_enabled": False,
        "topk_cascade_enabled": False,
        "proxy_model": "embedding",
        "recall_target": 0.9,
        "precision_target": 0.9,
        "failure_probability": 0.2,
        "sampling_percentage": 0.2,
        "min_join_cascade_size": 100,
    },
    "pipelines": {
        "mem0": {
            "max_messages": 40,
            "sim_noop_threshold": 0.92,
            "main_lm_overrides": {},
            "helper_lm_overrides": {},
        },
        "zep": {
            "max_messages": 40,
            "sliding_window_size": 10,
            "auto_build_communities_after_msg": 6,
            "main_lm_overrides": {},
            "helper_lm_overrides": {},
        },
        "evermemos": {
            "max_messages": 40,
            "boundary_msg_threshold": 15,
            "topic_sim_threshold": 0.5,
            "topic_max_gap_days": 7,
            "profile_min_segments": 2,
            "main_lm_overrides": {
                "max_tokens": 1024,
                "max_batch_size": 4,
            },
            "helper_lm_overrides": {
                "max_tokens": 1024,
                "max_batch_size": 4,
            },
        },
    },
}


@dataclass(frozen=True)
class GlobalConfig:
    locomo_path: Path
    conv_index: int
    enable_cache: bool


@dataclass(frozen=True)
class LMConfig:
    model: str
    max_tokens: int
    temperature: float
    max_batch_size: int
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HelperLMConfig:
    enabled: bool
    model: str
    max_tokens: int
    temperature: float
    max_batch_size: int
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RMConfig:
    model: str


@dataclass(frozen=True)
class OptimizationsConfig:
    filter_cascade_enabled: bool
    join_cascade_enabled: bool
    topk_cascade_enabled: bool
    proxy_model: str
    recall_target: float
    precision_target: float
    failure_probability: float
    sampling_percentage: float
    min_join_cascade_size: int


@dataclass(frozen=True)
class Mem0PipelineConfig:
    max_messages: int
    sim_noop_threshold: float
    main_lm_overrides: dict[str, Any] = field(default_factory=dict)
    helper_lm_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ZepPipelineConfig:
    max_messages: int
    sliding_window_size: int
    auto_build_communities_after_msg: int
    main_lm_overrides: dict[str, Any] = field(default_factory=dict)
    helper_lm_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvermemosPipelineConfig:
    max_messages: int
    boundary_msg_threshold: int
    topic_sim_threshold: float
    topic_max_gap_days: int
    profile_min_segments: int
    main_lm_overrides: dict[str, Any] = field(default_factory=dict)
    helper_lm_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationConfig:
    config_path: Path
    global_config: GlobalConfig
    main_model: LMConfig
    helper_model: HelperLMConfig
    rm_model: RMConfig
    optimizations: OptimizationsConfig
    mem0: Mem0PipelineConfig
    zep: ZepPipelineConfig
    evermemos: EvermemosPipelineConfig
    raw_config: dict[str, Any]


PipelineConfig = Mem0PipelineConfig | ZepPipelineConfig | EvermemosPipelineConfig


def _default_config_path() -> Path:
    return Path(__file__).resolve().with_name("config.yaml")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_dict(value: Any, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return copy.deepcopy(default) if default is not None else {}


def _build_lm_config(cfg: dict[str, Any]) -> LMConfig:
    return LMConfig(
        model=str(cfg.get("model", "")),
        max_tokens=int(cfg.get("max_tokens", 1000)),
        temperature=float(cfg.get("temperature", 0.0)),
        max_batch_size=int(cfg.get("max_batch_size", 2)),
        kwargs=_to_dict(cfg.get("kwargs"), {}),
    )


def _build_helper_lm_config(cfg: dict[str, Any]) -> HelperLMConfig:
    return HelperLMConfig(
        enabled=bool(cfg.get("enabled", False)),
        model=str(cfg.get("model", "")),
        max_tokens=int(cfg.get("max_tokens", 1000)),
        temperature=float(cfg.get("temperature", 0.0)),
        max_batch_size=int(cfg.get("max_batch_size", 2)),
        kwargs=_to_dict(cfg.get("kwargs"), {}),
    )


def _apply_model_overrides(model_cfg: LMConfig, overrides: dict[str, Any]) -> LMConfig:
    merged_kwargs = dict(model_cfg.kwargs)
    merged_kwargs.update(_to_dict(overrides.get("kwargs"), {}))
    return LMConfig(
        model=str(overrides.get("model", model_cfg.model)),
        max_tokens=int(overrides.get("max_tokens", model_cfg.max_tokens)),
        temperature=float(overrides.get("temperature", model_cfg.temperature)),
        max_batch_size=int(overrides.get("max_batch_size", model_cfg.max_batch_size)),
        kwargs=merged_kwargs,
    )


def _apply_helper_model_overrides(
    helper_cfg: HelperLMConfig, overrides: dict[str, Any]
) -> HelperLMConfig:
    merged_kwargs = dict(helper_cfg.kwargs)
    merged_kwargs.update(_to_dict(overrides.get("kwargs"), {}))
    return HelperLMConfig(
        enabled=bool(overrides.get("enabled", helper_cfg.enabled)),
        model=str(overrides.get("model", helper_cfg.model)),
        max_tokens=int(overrides.get("max_tokens", helper_cfg.max_tokens)),
        temperature=float(overrides.get("temperature", helper_cfg.temperature)),
        max_batch_size=int(overrides.get("max_batch_size", helper_cfg.max_batch_size)),
        kwargs=merged_kwargs,
    )


def load_sim_config() -> SimulationConfig:
    config_path = Path(
        os.getenv("LOTUS_SIM_CONFIG_PATH", str(_default_config_path()))
    ).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Simulation config not found at: {config_path}. "
            "Set LOTUS_SIM_CONFIG_PATH or create lotus-simulation/config.yaml."
        )

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid YAML root in {config_path}; expected a mapping.")

    merged = _deep_merge(DEFAULT_SIM_CONFIG, loaded)

    global_cfg = _to_dict(merged.get("global"), {})
    models_cfg = _to_dict(merged.get("models"), {})
    optim_cfg = _to_dict(merged.get("optimizations"), {})
    pipelines_cfg = _to_dict(merged.get("pipelines"), {})

    locomo_path = Path(str(global_cfg.get("locomo_path", ""))).expanduser()
    if not locomo_path.is_absolute():
        locomo_path = (config_path.parent / locomo_path).resolve()

    sim_config = SimulationConfig(
        config_path=config_path,
        global_config=GlobalConfig(
            locomo_path=locomo_path,
            conv_index=int(global_cfg.get("conv_index", 0)),
            enable_cache=bool(global_cfg.get("enable_cache", True)),
        ),
        main_model=_build_lm_config(_to_dict(models_cfg.get("main"), {})),
        helper_model=_build_helper_lm_config(_to_dict(models_cfg.get("helper"), {})),
        rm_model=RMConfig(model=str(_to_dict(models_cfg.get("rm"), {}).get("model", "intfloat/e5-base-v2"))),
        optimizations=OptimizationsConfig(
            filter_cascade_enabled=bool(optim_cfg.get("filter_cascade_enabled", False)),
            join_cascade_enabled=bool(optim_cfg.get("join_cascade_enabled", False)),
            topk_cascade_enabled=bool(optim_cfg.get("topk_cascade_enabled", False)),
            proxy_model=str(optim_cfg.get("proxy_model", "embedding")),
            recall_target=float(optim_cfg.get("recall_target", 0.9)),
            precision_target=float(optim_cfg.get("precision_target", 0.9)),
            failure_probability=float(optim_cfg.get("failure_probability", 0.2)),
            sampling_percentage=float(optim_cfg.get("sampling_percentage", 0.2)),
            min_join_cascade_size=int(optim_cfg.get("min_join_cascade_size", 100)),
        ),
        mem0=Mem0PipelineConfig(**_to_dict(pipelines_cfg.get("mem0"), {})),
        zep=ZepPipelineConfig(**_to_dict(pipelines_cfg.get("zep"), {})),
        evermemos=EvermemosPipelineConfig(**_to_dict(pipelines_cfg.get("evermemos"), {})),
        raw_config=merged,
    )
    return sim_config


def get_pipeline_settings(config: SimulationConfig, pipeline: str) -> PipelineConfig:
    if pipeline == "mem0":
        return config.mem0
    if pipeline == "zep":
        return config.zep
    if pipeline == "evermemos":
        return config.evermemos
    raise ValueError(f"Unknown pipeline: {pipeline}")


def get_pipeline_main_model(config: SimulationConfig, pipeline: str) -> LMConfig:
    pipeline_cfg = get_pipeline_settings(config, pipeline)
    return _apply_model_overrides(config.main_model, pipeline_cfg.main_lm_overrides)


def get_pipeline_helper_model(config: SimulationConfig, pipeline: str) -> HelperLMConfig:
    pipeline_cfg = get_pipeline_settings(config, pipeline)
    return _apply_helper_model_overrides(
        config.helper_model, pipeline_cfg.helper_lm_overrides
    )


def _path_to_str(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _path_to_str(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_path_to_str(v) for v in value]
    return value


def build_resolved_config_snapshot(
    config: SimulationConfig, pipeline: str
) -> dict[str, Any]:
    pipeline_cfg = get_pipeline_settings(config, pipeline)
    snapshot = {
        "global": asdict(config.global_config),
        "models": {
            "main": asdict(config.main_model),
            "helper": asdict(config.helper_model),
            "rm": asdict(config.rm_model),
        },
        "optimizations": asdict(config.optimizations),
        "pipeline_name": pipeline,
        "pipeline": asdict(pipeline_cfg),
        "effective_models": {
            "main": asdict(get_pipeline_main_model(config, pipeline)),
            "helper": asdict(get_pipeline_helper_model(config, pipeline)),
        },
    }
    return _path_to_str(snapshot)
