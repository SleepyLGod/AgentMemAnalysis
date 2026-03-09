import warnings
from dataclasses import dataclass

from lotus.types import CascadeArgs, ProxyModel


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


def _normalize_proxy_model(proxy_model: str) -> str:
    lowered = proxy_model.strip().lower()
    if lowered in {"embedding", "embedding_model"}:
        return "embedding"
    if lowered in {"helper_lm", "helper", "helper-model"}:
        return "helper_lm"
    warnings.warn(
        f"Invalid proxy_model={proxy_model!r}; falling back to 'embedding'."
    )
    return "embedding"


def load_opt_config(
    *,
    filter_cascade_enabled: bool,
    join_cascade_enabled: bool,
    topk_cascade_enabled: bool,
    proxy_model: str,
    recall_target: float,
    precision_target: float,
    failure_probability: float,
    sampling_percentage: float,
    min_join_cascade_size: int,
    helper_model: str | None,
) -> LotusOptConfig:
    helper = helper_model.strip() if isinstance(helper_model, str) else ""
    return LotusOptConfig(
        filter_cascade_enabled=bool(filter_cascade_enabled),
        join_cascade_enabled=bool(join_cascade_enabled),
        topk_cascade_enabled=bool(topk_cascade_enabled),
        proxy_model=_normalize_proxy_model(proxy_model),
        recall_target=float(recall_target),
        precision_target=float(precision_target),
        failure_probability=float(failure_probability),
        sampling_percentage=float(sampling_percentage),
        min_join_cascade_size=int(min_join_cascade_size),
        helper_model=helper if helper else None,
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
