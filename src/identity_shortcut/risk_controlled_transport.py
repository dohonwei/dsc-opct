from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata


ACTION_APPLICABILITY_FEATURES = (
    "relative_mean_gain",
    "relative_covariance_gain",
    "relative_swd_gain",
    "geometry_ratio",
    "probability_mean_shift",
    "probability_rank",
    "decision_flip_rate",
)


@dataclass(frozen=True)
class RiskControlledConfig:
    configuration_budgets: tuple[int, ...] = (16, 32, 64)
    configurations_per_cluster: int = 4
    familywise_alpha: float = 0.05
    bootstrap_repetitions: int = 5000
    auc_noninferiority_margin: float = -0.02
    minimum_brier_gain: float = 0.0
    minimum_probability_rank: float = 0.95
    maximum_absolute_probability_mean_shift: float = 0.10
    maximum_absolute_predicted_brier_gain: float = 1.0
    maximum_absolute_predicted_auc_delta: float = 1.0
    minimum_valid_auc_bootstraps: int = 200
    cluster_columns: tuple[str, ...] = (
        "task",
        "representation",
        "model",
        "split_seed",
    )
    balance_columns: tuple[str, ...] = ("task", "representation", "model")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def cluster_budgets(self) -> tuple[int, ...]:
        output = []
        for budget in self.configuration_budgets:
            if budget % self.configurations_per_cluster:
                raise ValueError(
                    f"Configuration budget {budget} is not divisible by "
                    f"cluster size {self.configurations_per_cluster}"
                )
            output.append(budget // self.configurations_per_cluster)
        return tuple(output)


def probability_methods(frame: pd.DataFrame) -> list[str]:
    methods = [
        column.removeprefix("probability_")
        for column in frame.columns
        if column.startswith("probability_") and not column.endswith("_oracle")
    ]
    if "identity" not in methods:
        raise ValueError("Prediction table lacks probability_identity")
    return sorted(set(methods), key=lambda method: (method != "identity", method))


def validate_prediction_contract(
    frame: pd.DataFrame,
    config: RiskControlledConfig,
) -> None:
    required = set(config.cluster_columns) | {"material_optimism_event", "nominal_dose"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction table lacks required columns: {missing}")
    methods = probability_methods(frame)
    probability_columns = [f"probability_{method}" for method in methods]
    values = frame[probability_columns].to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Candidate probabilities must be finite and lie in [0, 1]")
    labels = frame.material_optimism_event.to_numpy()
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("material_optimism_event must be binary")
    sizes = frame.groupby(list(config.cluster_columns), dropna=False).size()
    if not sizes.eq(config.configurations_per_cluster).all():
        observed = sizes.value_counts().sort_index().to_dict()
        raise ValueError(
            "Every audit cluster must contain exactly "
            f"{config.configurations_per_cluster} configurations; observed {observed}"
        )
    if frame.duplicated(list(config.cluster_columns) + ["nominal_dose"]).any():
        raise ValueError("Duplicate cluster-dose configurations detected")


def applicability_table(
    signatures: pd.DataFrame,
    candidate_methods: Iterable[str],
    config: RiskControlledConfig,
) -> pd.DataFrame:
    required = {"method", "probability_rank", "probability_mean_shift"}
    missing = sorted(required - set(signatures.columns))
    if missing:
        raise ValueError(f"Signature table lacks applicability columns: {missing}")
    signatures = signatures.drop_duplicates("method", keep="last").set_index("method")
    rows = []
    for method in candidate_methods:
        if method == "identity":
            rows.append(
                {
                    "method": method,
                    "applicable": True,
                    "applicability_reason": "identity_reference",
                    "probability_rank": 1.0,
                    "probability_mean_shift": 0.0,
                    "bounded_meta_prediction": True,
                }
            )
            continue
        if method not in signatures.index:
            rows.append(
                {
                    "method": method,
                    "applicable": False,
                    "applicability_reason": "missing_unlabeled_signature",
                    "probability_rank": np.nan,
                    "probability_mean_shift": np.nan,
                    "bounded_meta_prediction": False,
                }
            )
            continue
        row = signatures.loc[method]
        rank_ok = bool(float(row.probability_rank) >= config.minimum_probability_rank)
        shift_ok = bool(
            abs(float(row.probability_mean_shift))
            <= config.maximum_absolute_probability_mean_shift
        )
        bounded = True
        if "predicted_brier_gain" in row.index:
            bounded &= bool(
                abs(float(row.predicted_brier_gain))
                <= config.maximum_absolute_predicted_brier_gain
            )
        if "predicted_auc_delta" in row.index:
            bounded &= bool(
                abs(float(row.predicted_auc_delta))
                <= config.maximum_absolute_predicted_auc_delta
            )
        conformal_ok = bool(
            "conformal_p_ood" not in row.index or float(row.conformal_p_ood) >= 0.05
        )
        reasons = []
        if not rank_ok:
            reasons.append("rank_below_floor")
        if not shift_ok:
            reasons.append("probability_mean_shift_exceeds_guard")
        if not bounded:
            reasons.append("impossible_meta_prediction")
        if not conformal_ok:
            reasons.append("conformal_ood")
        rows.append(
            {
                "method": method,
                "applicable": bool(rank_ok and shift_ok and bounded and conformal_ok),
                "applicability_reason": "passed" if not reasons else ";".join(reasons),
                "probability_rank": float(row.probability_rank),
                "probability_mean_shift": float(row.probability_mean_shift),
                "bounded_meta_prediction": bool(bounded),
                "conformal_p_ood": (
                    float(row.conformal_p_ood)
                    if "conformal_p_ood" in row.index
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def attach_conformal_ood(
    development: pd.DataFrame,
    target: pd.DataFrame,
    neighbors: int = 5,
    features: Sequence[str] = ACTION_APPLICABILITY_FEATURES,
) -> pd.DataFrame:
    missing = sorted(
        ({"method"} | set(features))
        - set(development.columns).intersection(target.columns)
    )
    if missing:
        raise ValueError(f"Conformal OOD inputs lack common columns: {missing}")
    output = target.copy()
    output["conformal_p_ood"] = np.nan
    for method, target_rows in output.groupby("method", sort=False):
        reference = development.loc[development.method == method]
        if len(reference) <= neighbors:
            continue
        scaler = StandardScaler().fit(reference.loc[:, features])
        reference_values = scaler.transform(reference.loc[:, features])
        target_values = scaler.transform(target_rows.loc[:, features])
        reference_knn = NearestNeighbors(n_neighbors=neighbors + 1).fit(reference_values)
        reference_distance = reference_knn.kneighbors(reference_values)[0][:, -1]
        target_knn = NearestNeighbors(n_neighbors=neighbors).fit(reference_values)
        target_distance = target_knn.kneighbors(target_values)[0][:, -1]
        p_values = np.asarray(
            [
                (1.0 + np.sum(reference_distance >= distance))
                / (len(reference_distance) + 1.0)
                for distance in target_distance
            ]
        )
        output.loc[target_rows.index, "conformal_p_ood"] = p_values
    return output


def balanced_cluster_order(
    frame: pd.DataFrame,
    config: RiskControlledConfig,
    seed: int,
) -> list[tuple[object, ...]]:
    clusters = frame.loc[:, list(config.cluster_columns)].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(clusters))
    remaining = set(range(len(clusters)))
    counts = {
        column: {value: 0 for value in clusters[column].drop_duplicates().tolist()}
        for column in config.balance_columns
    }
    availability = {
        column: clusters[column].value_counts(dropna=False).to_dict()
        for column in config.balance_columns
    }
    order: list[tuple[object, ...]] = []
    while remaining:

        def score(index: int) -> tuple[float, float]:
            row = clusters.iloc[index]
            imbalance = sum(
                (counts[column][row[column]] + 1.0)
                / max(float(availability[column][row[column]]), 1.0)
                for column in config.balance_columns
            )
            return imbalance, float(jitter[index])

        selected = min(remaining, key=score)
        row = clusters.iloc[selected]
        key = tuple(row[column] for column in config.cluster_columns)
        order.append(key)
        for column in config.balance_columns:
            counts[column][row[column]] += 1
        remaining.remove(selected)
    return order


def rows_for_clusters(
    frame: pd.DataFrame,
    cluster_columns: Sequence[str],
    clusters: Sequence[tuple[object, ...]],
) -> np.ndarray:
    wanted = set(clusters)
    keys = frame.loc[:, list(cluster_columns)].itertuples(index=False, name=None)
    return np.fromiter((key in wanted for key in keys), dtype=bool, count=len(frame))


def _safe_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return np.nan
    ranks = rankdata(probability, method="average")
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def paired_point_metrics(
    frame: pd.DataFrame,
    methods: Iterable[str],
) -> pd.DataFrame:
    labels = frame.material_optimism_event.to_numpy(int)
    identity = frame.probability_identity.to_numpy(float)
    identity_loss = (labels - identity) ** 2
    identity_auc = _safe_auc(labels, identity)
    rows = []
    for method in methods:
        probability = frame[f"probability_{method}"].to_numpy(float)
        gain = float(np.mean(identity_loss - (labels - probability) ** 2))
        auc = _safe_auc(labels, probability)
        rows.append(
            {
                "method": method,
                "brier_gain": gain,
                "auc_delta": (
                    auc - identity_auc
                    if np.isfinite(auc) and np.isfinite(identity_auc)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def certify_checkpoint(
    audit: pd.DataFrame,
    candidate_methods: Sequence[str],
    config: RiskControlledConfig,
    seed: int,
    total_registered_candidates: int | None = None,
) -> pd.DataFrame:
    if not candidate_methods:
        return pd.DataFrame()
    reset = audit.reset_index(drop=True)
    cluster_groups = [
        group.index.to_numpy(int)
        for _, group in reset.groupby(list(config.cluster_columns), sort=False, dropna=False)
    ]
    rng = np.random.default_rng(seed)
    repetitions = config.bootstrap_repetitions
    labels = reset.material_optimism_event.to_numpy(int)
    identity = reset.probability_identity.to_numpy(float)
    identity_loss = (labels - identity) ** 2
    candidate_probability = {
        method: reset[f"probability_{method}"].to_numpy(float)
        for method in candidate_methods
    }
    point = paired_point_metrics(reset, candidate_methods).set_index("method")
    draws = {method: np.empty(repetitions, dtype=float) for method in candidate_methods}
    auc_draws = {
        method: np.full(repetitions, np.nan, dtype=float) for method in candidate_methods
    }
    for repetition in range(repetitions):
        sampled_groups = rng.integers(0, len(cluster_groups), size=len(cluster_groups))
        indices = np.concatenate([cluster_groups[index] for index in sampled_groups])
        sampled_labels = labels[indices]
        identity_auc = _safe_auc(sampled_labels, identity[indices])
        for method in candidate_methods:
            probability = candidate_probability[method]
            draws[method][repetition] = float(
                np.mean(identity_loss[indices] - (sampled_labels - probability[indices]) ** 2)
            )
            candidate_auc = _safe_auc(sampled_labels, probability[indices])
            if np.isfinite(identity_auc) and np.isfinite(candidate_auc):
                auc_draws[method][repetition] = candidate_auc - identity_auc
    family_candidates = total_registered_candidates or len(candidate_methods)
    family_size = max(1, family_candidates * len(config.configuration_budgets) * 2)
    tail_alpha = config.familywise_alpha / family_size
    rows = []
    for method in candidate_methods:
        valid_auc = auc_draws[method][np.isfinite(auc_draws[method])]
        brier_lcb = float(np.quantile(draws[method], tail_alpha))
        auc_lcb = (
            float(np.quantile(valid_auc, tail_alpha))
            if len(valid_auc) >= config.minimum_valid_auc_bootstraps
            else np.nan
        )
        brier_pass = brier_lcb > config.minimum_brier_gain
        auc_pass = bool(
            np.isfinite(auc_lcb) and auc_lcb >= config.auc_noninferiority_margin
        )
        rows.append(
            {
                "method": method,
                "brier_gain": float(point.loc[method, "brier_gain"]),
                "brier_gain_lcb": brier_lcb,
                "auc_delta": float(point.loc[method, "auc_delta"]),
                "auc_delta_lcb": auc_lcb,
                "tail_alpha": tail_alpha,
                "bootstrap_repetitions": repetitions,
                "valid_auc_bootstraps": int(len(valid_auc)),
                "brier_certified": bool(brier_pass),
                "auc_noninferior": bool(auc_pass),
                "certified": bool(brier_pass and auc_pass),
                "failure_reason": (
                    "certified"
                    if brier_pass and auc_pass
                    else (
                        "insufficient_two_class_auc_bootstraps"
                        if len(valid_auc) < config.minimum_valid_auc_bootstraps
                        else (
                            "brier_lower_bound_not_positive"
                            if not brier_pass
                            else "auc_noninferiority_not_certified"
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def select_certified_action(certification: pd.DataFrame) -> tuple[str, str]:
    if certification.empty:
        return "identity", "abstain_no_applicable_candidate"
    eligible = certification.loc[certification.certified]
    if eligible.empty:
        if certification.valid_auc_bootstraps.max() == 0:
            return "identity", "abstain_audit_contains_one_outcome_class"
        return "identity", "abstain_no_simultaneous_certificate"
    selected = eligible.sort_values(
        ["brier_gain_lcb", "brier_gain", "method"],
        ascending=[False, False, True],
    ).iloc[0]
    return str(selected.method), "selected_simultaneously_certified_adapter"


def run_registered_checkpoints(
    frame: pd.DataFrame,
    signatures: pd.DataFrame,
    config: RiskControlledConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[object, ...]]]:
    validate_prediction_contract(frame, config)
    methods = probability_methods(frame)
    applicability = applicability_table(signatures, methods, config)
    applicable = applicability.loc[
        applicability.applicable & applicability.method.ne("identity"), "method"
    ].tolist()
    order = balanced_cluster_order(frame, config, seed)
    if max(config.cluster_budgets) >= len(order):
        raise ValueError("Largest audit budget must leave at least one held-out cluster")
    decisions = []
    certificates = []
    registered_candidates = max(1, len([method for method in methods if method != "identity"]))
    locked_method: str | None = None
    locked_configuration_budget: int | None = None
    locked_cluster_budget: int | None = None
    for configuration_budget, cluster_budget in zip(
        config.configuration_budgets, config.cluster_budgets, strict=True
    ):
        if locked_method is None:
            audit_clusters = order[:cluster_budget]
            audit_mask = rows_for_clusters(frame, config.cluster_columns, audit_clusters)
            audit = frame.loc[audit_mask].reset_index(drop=True)
            certification = certify_checkpoint(
                audit,
                applicable,
                config,
                seed + configuration_budget * 1009,
                total_registered_candidates=registered_candidates,
            )
            selected, reason = select_certified_action(certification)
            if selected != "identity":
                locked_method = selected
                locked_configuration_budget = configuration_budget
                locked_cluster_budget = cluster_budget
        else:
            selected = locked_method
            reason = "carried_forward_locked_certificate"
            certification = pd.DataFrame()
            audit_clusters = order[: int(locked_cluster_budget)]
            audit_mask = rows_for_clusters(frame, config.cluster_columns, audit_clusters)
            audit = frame.loc[audit_mask].reset_index(drop=True)
        decisions.append(
            {
                "configuration_budget": configuration_budget,
                "cluster_budget": cluster_budget,
                "actual_configuration_budget": (
                    locked_configuration_budget
                    if locked_configuration_budget is not None
                    else configuration_budget
                ),
                "actual_cluster_budget": (
                    locked_cluster_budget if locked_cluster_budget is not None else cluster_budget
                ),
                "n_audit_rows": len(audit),
                "n_audit_events": int(audit.material_optimism_event.sum()),
                "n_applicable_candidates": len(applicable),
                "selected_method": selected,
                "selection_reason": reason,
            }
        )
        if not certification.empty:
            certification.insert(0, "configuration_budget", configuration_budget)
            certification.insert(1, "cluster_budget", cluster_budget)
            certificates.append(certification)
    certificate_table = (
        pd.concat(certificates, ignore_index=True) if certificates else pd.DataFrame()
    )
    return pd.DataFrame(decisions), certificate_table, order
