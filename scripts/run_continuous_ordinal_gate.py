from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


REPRESENTATIONS = {
    "stimulus_only": "stimulus__",
    "baseline_referenced": "baseline__",
    "phase_aware": "phase__",
}


class CumulativeOrdinalRegressor(BaseEstimator, RegressorMixin):
    """Nine-point ordinal prediction via eight cumulative logistic models."""

    def __init__(self, C: float = 1.0) -> None:
        self.C = C

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CumulativeOrdinalRegressor":
        self.models_ = []
        for threshold in range(1, 9):
            target = (y > threshold).astype(int)
            if np.unique(target).size < 2:
                self.models_.append(float(target[0]))
            else:
                model = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
                model.fit(x, target)
                self.models_.append(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        cumulative = []
        for model in self.models_:
            if isinstance(model, float):
                cumulative.append(np.full(x.shape[0], model))
            else:
                cumulative.append(model.predict_proba(x)[:, 1])
        probabilities = np.minimum.accumulate(np.stack(cumulative, axis=1), axis=1)
        return 1.0 + probabilities.sum(axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous and ordinal target gate under strict LOSO.")
    parser.add_argument("--feature-file", type=Path, default=Path("outputs/baseline_value_gate/trial_features.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/continuous_ordinal_gate"))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def concordance(y: np.ndarray, prediction: np.ndarray) -> float:
    covariance = np.mean((y - y.mean()) * (prediction - prediction.mean()))
    denominator = y.var() + prediction.var() + (y.mean() - prediction.mean()) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else np.nan


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    clipped = np.clip(prediction, 1, 9)
    binary_true = (y >= 5).astype(int)
    binary_prediction = (clipped >= 5).astype(int)
    rho = stats.spearmanr(y, clipped).statistic if np.unique(clipped).size > 1 else np.nan
    return {
        "mae": float(mean_absolute_error(y, clipped)),
        "rmse": float(mean_squared_error(y, clipped) ** 0.5),
        "spearman": float(rho),
        "ccc": concordance(y, clipped),
        "derived_balanced_accuracy": float(balanced_accuracy_score(binary_true, binary_prediction)),
    }


def model_space(target_type: str) -> tuple[Pipeline, dict[str, list[float]]]:
    if target_type == "continuous":
        model = Ridge()
        grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    elif target_type == "ordinal":
        model = CumulativeOrdinalRegressor()
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    elif target_type == "binary":
        model = LogisticRegression(max_iter=3000, class_weight="balanced")
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    else:
        raise ValueError(target_type)
    return Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler()), ("model", model)]), grid


def run(frame: pd.DataFrame, tasks: list[str], n_jobs: int) -> pd.DataFrame:
    subjects = frame.subject_id.unique()
    target_types = ("binary", "ordinal", "continuous")
    rows = []
    total = len(subjects) * len(tasks) * len(REPRESENTATIONS) * len(target_types)
    progress = tqdm(total=total, desc="Continuous/ordinal LOSO gate", unit="fit", dynamic_ncols=True)
    for task in tasks:
        score_column = f"{task}_score"
        for held_out in subjects:
            train_mask = frame.subject_id != held_out
            test_mask = ~train_mask
            groups = frame.loc[train_mask, "subject_id"].to_numpy()
            cv = GroupKFold(3)
            y_score_train = frame.loc[train_mask, score_column].to_numpy()
            y_score_test = frame.loc[test_mask, score_column].to_numpy()
            for representation, prefix in REPRESENTATIONS.items():
                columns = [column for column in frame if column.startswith(prefix)]
                train_x = frame.loc[train_mask, columns].to_numpy()
                test_x = frame.loc[test_mask, columns].to_numpy()
                for target_type in target_types:
                    pipeline, grid = model_space(target_type)
                    if target_type == "binary":
                        train_y = (y_score_train >= 5).astype(int)
                        scoring = "balanced_accuracy"
                    else:
                        train_y = y_score_train.astype(int) if target_type == "ordinal" else y_score_train
                        scoring = "neg_mean_absolute_error"
                    search = GridSearchCV(
                        pipeline, grid, scoring=scoring, cv=cv, n_jobs=n_jobs, refit=True, error_score="raise"
                    )
                    search.fit(train_x, train_y, groups=groups)
                    if target_type == "binary":
                        probability = search.best_estimator_.predict_proba(test_x)[:, 1]
                        prediction = 1 + 8 * probability
                    else:
                        prediction = search.best_estimator_.predict(test_x)
                    rows.append(
                        {
                            "task": task,
                            "held_out_subject": held_out,
                            "representation": representation,
                            "target_type": target_type,
                            "n_features": len(columns),
                            "inner_best_score": search.best_score_,
                            "best_params": str(search.best_params_),
                            **metrics(y_score_test, prediction),
                        }
                    )
                    progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def paired_target_tests(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, representation), group in results.groupby(["task", "representation"]):
        for metric, higher_is_better in (("mae", False), ("spearman", True), ("ccc", True), ("derived_balanced_accuracy", True)):
            wide = group.pivot(index="held_out_subject", columns="target_type", values=metric)
            for left, right in (("ordinal", "binary"), ("continuous", "binary"), ("ordinal", "continuous")):
                pair = wide[[left, right]].dropna()
                delta = pair[left] - pair[right]
                oriented = delta if higher_is_better else -delta
                statistic, p_value = stats.wilcoxon(oriented, zero_method="zsplit", mode="approx") if np.any(oriented != 0) else (np.nan, np.nan)
                rows.append(
                    {
                        "task": task,
                        "representation": representation,
                        "metric": metric,
                        "left": left,
                        "right": right,
                        "n_subjects": len(pair),
                        "mean_improvement": float(oriented.mean()),
                        "wilcoxon_statistic": statistic,
                        "p_raw": p_value,
                    }
                )
    output = pd.DataFrame(rows)
    output["p_holm_within_task"] = np.nan
    for _, indices in output.groupby("task").groups.items():
        indices = np.asarray(indices)
        p = output.loc[indices, "p_raw"].to_numpy()
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        running = 0.0
        for rank, position in enumerate(order):
            running = max(running, (len(p) - rank) * p[position])
            adjusted[position] = min(running, 1.0)
        output.loc[indices, "p_holm_within_task"] = adjusted
    return output


def label_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in ("arousal", "valence"):
        values = frame[f"{task}_score"].to_numpy()
        subject_means = frame.groupby("subject_id")[f"{task}_score"].mean()
        rows.append(
            {
                "task": task,
                "n_trials": len(values),
                "near_boundary_4_to_6_fraction": float(np.mean((values >= 4) & (values <= 6))),
                "score_mean": float(values.mean()),
                "score_std": float(values.std()),
                "subject_mean_std": float(subject_means.std()),
                "subject_mean_range": float(subject_means.max() - subject_means.min()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.feature_file)
    label_audit(frame).to_csv(args.output_root / "label_audit.csv", index=False)
    results = run(frame, args.tasks, args.n_jobs)
    results.to_csv(args.output_root / "per_subject_results.csv", index=False)
    results.groupby(["task", "representation", "target_type"])[
        ["mae", "rmse", "spearman", "ccc", "derived_balanced_accuracy"]
    ].agg(["mean", "std", "count"]).to_csv(args.output_root / "summary.csv")
    paired_target_tests(results).to_csv(args.output_root / "paired_target_tests.csv", index=False)
    print(results.groupby(["task", "representation", "target_type"])[["mae", "spearman", "ccc", "derived_balanced_accuracy"]].mean().round(4))


if __name__ == "__main__":
    main()
