import os
import warnings
from dataclasses import dataclass

from lotus.types import CascadeArgs, ProxyModel


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    lowered = raw.lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    warnings.warn(f"Invalid boolean for {name}={raw!r}; using default={default}.")
    return default


def _parse_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(f"Invalid int for {name}={raw!r}; using default={default}.")
        return default


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.warn(f"Invalid float for {name}={raw!r}; using default={default}.")
        return default


def _parse_proxy_model(default: str = "embedding") -> str:
    raw = os.getenv("LOTUS_OPT_PROXY_MODEL", "").strip().lower()
    if raw == "":
        return default
    if raw in {"embedding", "embedding_model"}:
        return "embedding"
    if raw in {"helper_lm", "helper", "helper-model"}:
        return "helper_lm"
    warnings.warn(
        f"Invalid LOTUS_OPT_PROXY_MODEL={raw!r}; using default={default!r}."
    )
    return default


@dataclass(frozen=True)
class LotusOptConfig:
    filter_cascade_enabled: bool
    join_cascade_enabled: bool
    topk_cascade_enabled: bool
    proxy_model: str
    recall_target: float
    precision_target: float
    failure_probability: float
    sampling_percentage: float
    min_join_cascade_size: int
    helper_model: str | None


def load_opt_config() -> LotusOptConfig:
    helper_model_raw = os.getenv("LOTUS_HELPER_MODEL", "").strip()
    helper_model = helper_model_raw if helper_model_raw else None

    return LotusOptConfig(
        filter_cascade_enabled=_parse_bool("LOTUS_OPT_FILTER_CASCADE", False),
        join_cascade_enabled=_parse_bool("LOTUS_OPT_JOIN_CASCADE", False),
        topk_cascade_enabled=_parse_bool("LOTUS_OPT_TOPK_CASCADE", False),
        proxy_model=_parse_proxy_model("embedding"),
        recall_target=_parse_float("LOTUS_OPT_RECALL_TARGET", 0.9),
        precision_target=_parse_float("LOTUS_OPT_PRECISION_TARGET", 0.9),
        failure_probability=_parse_float("LOTUS_OPT_FAILURE_PROB", 0.2),
        sampling_percentage=_parse_float("LOTUS_OPT_SAMPLING_PCT", 0.2),
        min_join_cascade_size=_parse_int("LOTUS_OPT_MIN_JOIN_CASCADE_SIZE", 100),
        helper_model=helper_model,
    )


def build_cascade_args(config: LotusOptConfig, proxy_model_name: str) -> CascadeArgs:
    proxy = (
        ProxyModel.HELPER_LM
        if proxy_model_name == "helper_lm"
        else ProxyModel.EMBEDDING_MODEL
    )
    return CascadeArgs(
        recall_target=config.recall_target,
        precision_target=config.precision_target,
        sampling_percentage=config.sampling_percentage,
        failure_probability=config.failure_probability,
        proxy_model=proxy,
        min_join_cascade_size=config.min_join_cascade_size,
    )
