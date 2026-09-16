from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from scipy import stats


EPS = 1e-6


def _sym_power(matrix: np.ndarray, power: float, ridge: float = 1e-4) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    values = np.maximum(values, ridge)
    return (vectors * values**power) @ vectors.T


def mean_shift(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return target - target.mean(axis=0) + source.mean(axis=0)


def coral(source: np.ndarray, target: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    source_center = source - source.mean(axis=0)
    target_center = target - target.mean(axis=0)
    source_cov = np.cov(source_center, rowvar=False) + ridge * np.eye(source.shape[1])
    target_cov = np.cov(target_center, rowvar=False) + ridge * np.eye(target.shape[1])
    transform = _sym_power(target_cov, -0.5, ridge) @ _sym_power(source_cov, 0.5, ridge)
    return target_center @ transform + source.mean(axis=0)


def quantile_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    mapped = np.empty_like(target, dtype=float)
    quantiles = np.linspace(0.0, 1.0, max(len(source), len(target)))
    for column in range(target.shape[1]):
        source_q = np.quantile(source[:, column], quantiles)
        target_q = np.quantile(target[:, column], quantiles)
        mapped[:, column] = np.interp(target[:, column], target_q, source_q)
    return mapped


def support_clip(source: np.ndarray, target: np.ndarray, quantile: float = 0.025) -> np.ndarray:
    lower = np.quantile(source, quantile, axis=0)
    upper = np.quantile(source, 1.0 - quantile, axis=0)
    return np.clip(target, lower, upper)


def covariance_distance(source: np.ndarray, target: np.ndarray) -> float:
    source_cov = np.cov(source, rowvar=False)
    target_cov = np.cov(target, rowvar=False)
    return float(np.linalg.norm(source_cov - target_cov, ord="fro"))


def mean_distance(source: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(source.mean(axis=0) - target.mean(axis=0)))


def sliced_wasserstein(
    source: np.ndarray,
    target: np.ndarray,
    projections: int = 64,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, source.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + EPS
    source_projected = source @ directions.T
    target_projected = target @ directions.T
    grid = np.linspace(0.0, 1.0, min(200, max(len(source), len(target))))
    values = []
    for column in range(projections):
        source_q = np.quantile(source_projected[:, column], grid)
        target_q = np.quantile(target_projected[:, column], grid)
        values.append(np.mean(np.abs(source_q - target_q)))
    return float(np.mean(values))


def discrepancy(source: np.ndarray, target: np.ndarray, seed: int = 0) -> dict[str, float]:
    return {
        "mean_distance": mean_distance(source, target),
        "covariance_distance": covariance_distance(source, target),
        "sliced_wasserstein": sliced_wasserstein(source, target, seed=seed),
    }


def support_violation(source: np.ndarray, target: np.ndarray, quantile: float = 0.025) -> float:
    lower = np.quantile(source, quantile, axis=0)
    upper = np.quantile(source, 1.0 - quantile, axis=0)
    return float(np.mean(np.any((target < lower) | (target > upper), axis=1)))


def pairwise_geometry_ratio(before: np.ndarray, after: np.ndarray) -> float:
    before_distance = np.linalg.norm(before[:, None] - before[None, :], axis=2)
    after_distance = np.linalg.norm(after[:, None] - after[None, :], axis=2)
    upper = np.triu_indices(len(before), 1)
    denominator = float(np.mean(before_distance[upper]))
    return float(np.mean(after_distance[upper]) / max(denominator, EPS))


@dataclass(frozen=True)
class SCMTConfig:
    hidden_width: int = 16
    epochs: int = 300
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    projections: int = 32
    risk_strata: int = 3
    lambda_covariance: float = 0.30
    lambda_geometry: float = 0.20
    lambda_identity: float = 0.35
    lambda_rank: float = 0.30
    lambda_displacement: float = 0.08
    lambda_support: float = 0.15
    gate_floor: float = 0.15
    gate_temperature: float = 0.35
    min_relative_alignment_gain: float = 0.08
    min_prediction_rank: float = 0.95
    geometry_ratio_low: float = 0.75
    geometry_ratio_high: float = 1.25
    max_mean_displacement: float = 1.25
    trust_region_projection: bool = True

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class ResidualTransport(torch.nn.Module):
    def __init__(self, dimensions: int, hidden_width: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dimensions, hidden_width),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_width, hidden_width),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_width, dimensions),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, values: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return values + gate * self.network(values)


def _support_gate(source: np.ndarray, target: np.ndarray, config: SCMTConfig) -> np.ndarray:
    center = np.mean(source, axis=0)
    covariance = np.cov(source, rowvar=False) + 1e-3 * np.eye(source.shape[1])
    inverse = np.linalg.pinv(covariance)
    source_distance = np.sqrt(np.einsum("ij,jk,ik->i", source - center, inverse, source - center))
    target_distance = np.sqrt(np.einsum("ij,jk,ik->i", target - center, inverse, target - center))
    threshold = float(np.quantile(source_distance, 0.75))
    scale = max(float(np.quantile(source_distance, 0.90) - threshold), 0.25)
    logistic = 1.0 / (1.0 + np.exp(-(target_distance - threshold) / (scale * config.gate_temperature)))
    return (config.gate_floor + (1.0 - config.gate_floor) * logistic)[:, None]


def _torch_swd(source: torch.Tensor, target: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    count = min(len(source), len(target))
    if count < 4:
        return source.new_tensor(0.0)
    source_projection = source @ directions.T
    target_projection = target @ directions.T
    source_sorted = torch.sort(source_projection, dim=0).values
    target_sorted = torch.sort(target_projection, dim=0).values
    source_index = torch.linspace(0, len(source_sorted) - 1, count, device=source.device).long()
    target_index = torch.linspace(0, len(target_sorted) - 1, count, device=source.device).long()
    return torch.mean(torch.abs(source_sorted[source_index] - target_sorted[target_index]))


def _covariance_tensor(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(len(values) - 1, 1)


def _risk_masks(probability: np.ndarray, boundaries: np.ndarray) -> list[np.ndarray]:
    bins = np.digitize(probability, boundaries[1:-1], right=True)
    return [bins == index for index in range(len(boundaries) - 1)]


def fit_scmt(
    source: np.ndarray,
    target: np.ndarray,
    risk_probability: Callable[[np.ndarray], np.ndarray],
    config: SCMTConfig,
    seed: int,
    device: str,
    epoch_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SCMT requires CUDA; pass a CPU device only for debugging")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source_tensor = torch.as_tensor(source, device=device)
    target_tensor = torch.as_tensor(target, device=device)
    gate_tensor = torch.as_tensor(
        _support_gate(source, target, config), device=device, dtype=torch.float32
    )
    model = ResidualTransport(source.shape[1], config.hidden_width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 991)
    directions = torch.randn(
        config.projections, source.shape[1], generator=generator, device=device
    )
    directions = directions / (torch.linalg.norm(directions, dim=1, keepdim=True) + EPS)
    source_probability = risk_probability(source)
    target_probability = risk_probability(target)
    boundaries = np.quantile(source_probability, np.linspace(0, 1, config.risk_strata + 1))
    boundaries[0], boundaries[-1] = -np.inf, np.inf
    source_masks = _risk_masks(source_probability, boundaries)
    target_masks = _risk_masks(target_probability, boundaries)
    source_masks_tensor = [torch.as_tensor(mask, device=device) for mask in source_masks]
    target_masks_tensor = [torch.as_tensor(mask, device=device) for mask in target_masks]
    lower = torch.as_tensor(
        np.quantile(source, 0.025, axis=0), device=device, dtype=torch.float32
    )
    upper = torch.as_tensor(
        np.quantile(source, 0.975, axis=0), device=device, dtype=torch.float32
    )
    pair_generator = np.random.default_rng(seed + 313)
    pair_count = min(2048, max(128, len(target) * 4))
    pair_left = torch.as_tensor(pair_generator.integers(0, len(target), pair_count), device=device)
    pair_right = torch.as_tensor(pair_generator.integers(0, len(target), pair_count), device=device)
    frozen_weight = None
    frozen_bias = None
    # Recover the local logit direction numerically; this keeps fit_scmt independent of sklearn.
    origin = np.zeros((1, source.shape[1]), dtype=np.float32)
    base_logit = stats.logistic.ppf(np.clip(risk_probability(origin), EPS, 1 - EPS))[0]
    columns = np.eye(source.shape[1], dtype=np.float32)
    column_logits = stats.logistic.ppf(np.clip(risk_probability(columns), EPS, 1 - EPS))
    frozen_weight = torch.as_tensor(column_logits - base_logit, device=device, dtype=torch.float32)
    frozen_bias = torch.tensor(base_logit, device=device, dtype=torch.float32)
    last_losses: dict[str, float] = {}
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        transformed = model(target_tensor, gate_tensor)
        source_identity = model(source_tensor, torch.ones_like(source_tensor))
        marginal_swd = _torch_swd(source_tensor, transformed, directions)
        conditional = source_tensor.new_tensor(0.0)
        valid_strata = 0
        for source_mask, target_mask in zip(
            source_masks_tensor, target_masks_tensor, strict=True
        ):
            if source_mask.sum().item() >= 4 and target_mask.sum().item() >= 4:
                conditional = conditional + _torch_swd(
                    source_tensor[source_mask], transformed[target_mask], directions
                )
                valid_strata += 1
        alignment = 0.35 * marginal_swd + 0.65 * conditional / max(valid_strata, 1)
        covariance = torch.mean(
            (_covariance_tensor(source_tensor) - _covariance_tensor(transformed)) ** 2
        )
        before_distance = torch.linalg.norm(
            target_tensor[pair_left] - target_tensor[pair_right], dim=1
        )
        after_distance = torch.linalg.norm(
            transformed[pair_left] - transformed[pair_right], dim=1
        )
        geometry = torch.mean((after_distance - before_distance) ** 2)
        identity = torch.mean((source_identity - source_tensor) ** 2)
        transformed_logit = transformed @ frozen_weight + frozen_bias
        original_logit = target_tensor @ frozen_weight + frozen_bias
        rank = torch.mean(
            ((transformed_logit - transformed_logit.mean()) - (original_logit - original_logit.mean())) ** 2
        )
        displacement = torch.mean((transformed - target_tensor) ** 2)
        support = torch.mean(torch.relu(lower - transformed) ** 2 + torch.relu(transformed - upper) ** 2)
        loss = (
            alignment
            + config.lambda_covariance * covariance
            + config.lambda_geometry * geometry
            + config.lambda_identity * identity
            + config.lambda_rank * rank
            + config.lambda_displacement * displacement
            + config.lambda_support * support
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_losses = {
            "total": float(loss.detach().cpu()),
            "alignment": float(alignment.detach().cpu()),
            "covariance": float(covariance.detach().cpu()),
            "geometry": float(geometry.detach().cpu()),
            "identity": float(identity.detach().cpu()),
            "rank": float(rank.detach().cpu()),
            "displacement": float(displacement.detach().cpu()),
            "support": float(support.detach().cpu()),
        }
        if epoch_callback is not None:
            epoch_callback(epoch, last_losses)
    model.eval()
    with torch.no_grad():
        transformed = model(target_tensor, gate_tensor).cpu().numpy()
    candidate_mean_displacement = float(
        np.mean(np.linalg.norm(transformed - target, axis=1))
    )
    trust_region_scale = 1.0
    if (
        config.trust_region_projection
        and candidate_mean_displacement > config.max_mean_displacement
    ):
        trust_region_scale = config.max_mean_displacement / candidate_mean_displacement
        transformed = target + trust_region_scale * (transformed - target)
    raw_discrepancy = discrepancy(source, target, seed)
    transformed_discrepancy = discrepancy(source, transformed, seed)
    raw_swd = raw_discrepancy["sliced_wasserstein"]
    relative_gain = (raw_swd - transformed_discrepancy["sliced_wasserstein"]) / max(raw_swd, EPS)
    prediction_rank = stats.spearmanr(risk_probability(target), risk_probability(transformed)).statistic
    geometry_ratio = pairwise_geometry_ratio(target, transformed)
    mean_displacement = float(np.mean(np.linalg.norm(transformed - target, axis=1)))
    checks = {
        "alignment": relative_gain >= config.min_relative_alignment_gain,
        "prediction_rank": prediction_rank >= config.min_prediction_rank,
        "geometry": config.geometry_ratio_low <= geometry_ratio <= config.geometry_ratio_high,
        "displacement": mean_displacement <= config.max_mean_displacement + 1e-6,
    }
    applied = bool(all(checks.values()))
    output = transformed if applied else target.copy()
    diagnostics: dict[str, object] = {
        "applied": applied,
        "checks": checks,
        "relative_alignment_gain": float(relative_gain),
        "raw_discrepancy": raw_discrepancy,
        "transformed_discrepancy": transformed_discrepancy,
        "prediction_rank": float(prediction_rank),
        "geometry_ratio": geometry_ratio,
        "mean_displacement": mean_displacement,
        "candidate_mean_displacement": candidate_mean_displacement,
        "trust_region_scale": float(trust_region_scale),
        "candidate_discrepancy": transformed_discrepancy,
        "raw_support_violation": support_violation(source, target),
        "candidate_support_violation": support_violation(source, transformed),
        "final_losses": last_losses,
        "risk_stratum_counts_source": [int(mask.sum()) for mask in source_masks],
        "risk_stratum_counts_target": [int(mask.sum()) for mask in target_masks],
    }
    return output.astype(float), diagnostics
