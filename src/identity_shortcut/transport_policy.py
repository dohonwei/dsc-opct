from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .domain_transport import discrepancy, pairwise_geometry_ratio, support_violation


EPS = 1e-8
POLICY_FEATURES = (
    "raw_mean_distance",
    "raw_covariance_distance",
    "raw_sliced_wasserstein",
    "raw_support_violation",
    "after_mean_distance",
    "after_covariance_distance",
    "after_sliced_wasserstein",
    "after_support_violation",
    "relative_mean_gain",
    "relative_covariance_gain",
    "relative_swd_gain",
    "mean_displacement",
    "geometry_ratio",
    "probability_mean_before",
    "probability_std_before",
    "estimated_target_prevalence",
    "estimated_label_shift",
    "probability_mean_after",
    "probability_std_after",
    "probability_mean_shift",
    "probability_rank",
    "decision_flip_rate",
)


@dataclass(frozen=True)
class PolicyConfig:
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    lower_quantile: float = 0.25
    minimum_brier_gain: float = 0.0
    auc_noninferiority_margin: float = 0.02
    max_estimated_label_shift: float = 0.25
    ambiguity_support_floor: float = 0.50
    ambiguity_mean_ceiling: float = 0.80
    ambiguity_covariance_ceiling: float = 1.80

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _relative_gain(before: float, after: float) -> float:
    return float((before - after) / max(abs(before), EPS))


def _safe_rank(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= EPS or np.std(right) <= EPS:
        return 1.0 if np.allclose(left, right) else 0.0
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def unlabeled_signature(
    source: np.ndarray,
    source_labels: np.ndarray,
    raw_target: np.ndarray,
    transformed: np.ndarray,
    risk_probability: Callable[[np.ndarray], np.ndarray],
    decision_threshold: float,
    seed: int,
) -> dict[str, float]:
    source = np.asarray(source, dtype=float)
    raw_target = np.asarray(raw_target, dtype=float)
    transformed = np.asarray(transformed, dtype=float)
    raw = discrepancy(source, raw_target, seed)
    after = discrepancy(source, transformed, seed)
    probability_before = risk_probability(raw_target)
    probability_after = risk_probability(transformed)
    source_probability = risk_probability(source)
    source_labels = np.asarray(source_labels, dtype=int)
    source_prevalence = float(np.mean(source_labels))
    mean_negative = float(np.mean(source_probability[source_labels == 0]))
    mean_positive = float(np.mean(source_probability[source_labels == 1]))
    denominator = mean_positive - mean_negative
    estimated_prevalence = (
        source_prevalence
        if abs(denominator) <= EPS
        else float(
            np.clip(
                (np.mean(probability_before) - mean_negative) / denominator,
                0.0,
                1.0,
            )
        )
    )
    displacement = np.linalg.norm(transformed - raw_target, axis=1)
    return {
        "raw_mean_distance": raw["mean_distance"],
        "raw_covariance_distance": raw["covariance_distance"],
        "raw_sliced_wasserstein": raw["sliced_wasserstein"],
        "raw_support_violation": support_violation(source, raw_target),
        "after_mean_distance": after["mean_distance"],
        "after_covariance_distance": after["covariance_distance"],
        "after_sliced_wasserstein": after["sliced_wasserstein"],
        "after_support_violation": support_violation(source, transformed),
        "relative_mean_gain": _relative_gain(raw["mean_distance"], after["mean_distance"]),
        "relative_covariance_gain": _relative_gain(
            raw["covariance_distance"], after["covariance_distance"]
        ),
        "relative_swd_gain": _relative_gain(
            raw["sliced_wasserstein"], after["sliced_wasserstein"]
        ),
        "mean_displacement": float(np.mean(displacement)),
        "geometry_ratio": pairwise_geometry_ratio(raw_target, transformed),
        "probability_mean_before": float(np.mean(probability_before)),
        "probability_std_before": float(np.std(probability_before)),
        "estimated_target_prevalence": estimated_prevalence,
        "estimated_label_shift": abs(estimated_prevalence - source_prevalence),
        "probability_mean_after": float(np.mean(probability_after)),
        "probability_std_after": float(np.std(probability_after)),
        "probability_mean_shift": float(np.mean(probability_after - probability_before)),
        "probability_rank": _safe_rank(probability_before, probability_after),
        "decision_flip_rate": float(
            np.mean(
                (probability_before >= decision_threshold)
                != (probability_after >= decision_threshold)
            )
        ),
    }


def fit_ridge(table: pd.DataFrame, target: str, alpha: float):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(table.loc[:, POLICY_FEATURES], table[target])
    return model


def choose_alpha(
    table: pd.DataFrame,
    target: str,
    config: PolicyConfig,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    families = sorted(table.shift_family.unique())
    methods = sorted(table.method.unique())
    for alpha in config.alpha_grid:
        squared_errors: list[float] = []
        for family in families:
            training = table.loc[table.shift_family != family]
            held_out = table.loc[table.shift_family == family]
            for method in methods:
                train_method = training.loc[training.method == method]
                test_method = held_out.loc[held_out.method == method]
                if train_method.empty or test_method.empty:
                    continue
                model = fit_ridge(train_method, target, alpha)
                prediction = model.predict(test_method.loc[:, POLICY_FEATURES])
                squared_errors.extend((prediction - test_method[target].to_numpy()) ** 2)
        rows.append({
            "alpha": float(alpha),
            "losfo_mse": float(np.mean(squared_errors)),
            "n_predictions": int(len(squared_errors)),
        })
    scores = pd.DataFrame(rows).sort_values(["losfo_mse", "alpha"])
    return float(scores.iloc[0].alpha), scores


def fit_family_committee(
    table: pd.DataFrame,
    target: str,
    alpha: float,
) -> dict[str, list[object]]:
    committee: dict[str, list[object]] = {}
    families = sorted(table.shift_family.unique())
    for method in sorted(table.method.unique()):
        method_table = table.loc[table.method == method]
        models = []
        for family in families:
            training = method_table.loc[method_table.shift_family != family]
            if not training.empty:
                models.append(fit_ridge(training, target, alpha))
        if not models:
            models.append(fit_ridge(method_table, target, alpha))
        committee[method] = models
    return committee


def committee_distribution(
    committee: dict[str, list[object]],
    table: pd.DataFrame,
) -> np.ndarray:
    predictions = []
    for row_index, row in table.iterrows():
        models = committee[str(row.method)]
        frame = table.loc[[row_index], POLICY_FEATURES]
        predictions.append([float(model.predict(frame)[0]) for model in models])
    return np.asarray(predictions, dtype=float)


def select_actions(
    table: pd.DataFrame,
    gain_committee: dict[str, list[object]],
    auc_committee: dict[str, list[object]],
    config: PolicyConfig,
) -> pd.DataFrame:
    output = table.copy()
    candidate_mask = output.method != "identity"
    candidates = output.loc[candidate_mask]
    gain_predictions = committee_distribution(gain_committee, candidates)
    auc_predictions = committee_distribution(auc_committee, candidates)
    output["predicted_brier_gain"] = 0.0
    output["predicted_brier_gain_lcb"] = 0.0
    output["predicted_auc_delta"] = 0.0
    output["predicted_auc_delta_lcb"] = 0.0
    output.loc[candidate_mask, "predicted_brier_gain"] = gain_predictions.mean(axis=1)
    output.loc[candidate_mask, "predicted_brier_gain_lcb"] = np.quantile(
        gain_predictions, config.lower_quantile, axis=1
    )
    output.loc[candidate_mask, "predicted_auc_delta"] = auc_predictions.mean(axis=1)
    output.loc[candidate_mask, "predicted_auc_delta_lcb"] = np.quantile(
        auc_predictions, config.lower_quantile, axis=1
    )
    label_identifiable = (
        output.estimated_label_shift <= config.max_estimated_label_shift
    )
    local_support_ambiguity = (
        (output.raw_support_violation > config.ambiguity_support_floor)
        & (output.raw_mean_distance < config.ambiguity_mean_ceiling)
        & (
            output.raw_covariance_distance
            < config.ambiguity_covariance_ceiling
        )
    )
    identifiable = label_identifiable & ~local_support_ambiguity
    eligible = candidate_mask & identifiable & (
        (output.predicted_brier_gain_lcb > config.minimum_brier_gain)
        & (
            output.predicted_auc_delta_lcb
            >= -config.auc_noninferiority_margin
        )
    )
    output["policy_eligible"] = eligible
    output["policy_score"] = np.where(
        eligible, output.predicted_brier_gain_lcb, -np.inf
    )
    decisions = []
    for _, scenario in output.groupby("scenario_id", sort=False):
        eligible_candidates = scenario.loc[scenario.policy_eligible]
        if eligible_candidates.empty:
            selected_method = "identity"
            if not bool(
                scenario.estimated_label_shift.iloc[0]
                <= config.max_estimated_label_shift
            ):
                reason = "abstain_label_shift_ambiguity"
            elif bool(
                (scenario.raw_support_violation.iloc[0] > config.ambiguity_support_floor)
                and (scenario.raw_mean_distance.iloc[0] < config.ambiguity_mean_ceiling)
                and (
                    scenario.raw_covariance_distance.iloc[0]
                    < config.ambiguity_covariance_ceiling
                )
            ):
                reason = "abstain_local_support_ambiguity"
            else:
                reason = "abstain_no_positive_safe_lower_bound"
        else:
            selected_method = str(
                eligible_candidates.loc[eligible_candidates.policy_score.idxmax(), "method"]
            )
            reason = "selected_positive_safe_lower_bound"
        selected = scenario.loc[scenario.method == selected_method]
        if selected.empty:
            raise ValueError(f"Scenario {scenario.scenario_id.iloc[0]} lacks {selected_method}")
        row = selected.iloc[0].to_dict()
        row["selected_method"] = selected_method
        row["selection_reason"] = reason
        decisions.append(row)
    return pd.DataFrame(decisions)
