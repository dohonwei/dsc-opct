from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


REPRESENTATIONS = {
    "stimulus_only": "stimulus__",
    "baseline_referenced": "baseline__",
    "phase_aware": "phase__",
}
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test global mean, video prior, and subject-bias explanations.")
    parser.add_argument(
        "--feature-file", type=Path, default=Path("outputs/baseline_value_gate/trial_features.csv")
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/bias_prior_gate"))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    return parser.parse_args()


def make_model(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def trial_prior(train_trial: np.ndarray, train_y: np.ndarray, query_trial: np.ndarray) -> np.ndarray:
    global_mean = float(np.mean(train_y))
    means = {int(trial): float(np.mean(train_y[train_trial == trial])) for trial in np.unique(train_trial)}
    return np.asarray([means.get(int(trial), global_mean) for trial in query_trial], dtype=float)


def subject_center(y: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    centered = np.empty_like(y, dtype=float)
    for subject in np.unique(subjects):
        mask = subjects == subject
        centered[mask] = y[mask] - np.mean(y[mask])
    return centered


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    trials: np.ndarray,
    target_mode: str,
) -> float:
    losses = {alpha: [] for alpha in ALPHAS}
    for inner_train, inner_valid in GroupKFold(3).split(x, y, groups=subjects):
        for alpha in ALPHAS:
            if target_mode == "absolute":
                train_target = y[inner_train]
                validation_target = y[inner_valid]
                offset_train = np.zeros(len(inner_train))
                offset_valid = np.zeros(len(inner_valid))
            elif target_mode == "trial_residual":
                offset_train = trial_prior(
                    trials[inner_train], y[inner_train], trials[inner_train]
                )
                offset_valid = trial_prior(
                    trials[inner_train], y[inner_train], trials[inner_valid]
                )
                train_target = y[inner_train] - offset_train
                validation_target = y[inner_valid]
            elif target_mode == "subject_centered":
                offset_train = np.zeros(len(inner_train))
                offset_valid = np.zeros(len(inner_valid))
                train_target = subject_center(y[inner_train], subjects[inner_train])
                validation_target = subject_center(y[inner_valid], subjects[inner_valid])
            else:
                raise ValueError(target_mode)
            model = make_model(alpha).fit(x[inner_train], train_target)
            prediction = offset_valid + model.predict(x[inner_valid])
            losses[alpha].append(mean_absolute_error(validation_target, prediction))
    return min(ALPHAS, key=lambda alpha: (np.mean(losses[alpha]), alpha))


def concordance(y: np.ndarray, prediction: np.ndarray) -> float:
    covariance = np.mean((y - y.mean()) * (prediction - prediction.mean()))
    denominator = y.var() + prediction.var() + (y.mean() - prediction.mean()) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else np.nan


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 1, 9)
    rho = stats.spearmanr(y, prediction).statistic if np.unique(prediction).size > 1 else np.nan
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "ccc": concordance(y, prediction),
        "derived_balanced_accuracy": float(
            balanced_accuracy_score((y >= 5).astype(int), (prediction >= 5).astype(int))
        ),
    }


def relative_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = y - np.mean(y)
    rho = stats.spearmanr(target, prediction).statistic if np.unique(prediction).size > 1 else np.nan
    return {
        "relative_mae": float(mean_absolute_error(target, prediction)),
        "relative_rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "relative_spearman": float(rho) if np.isfinite(rho) else np.nan,
        "relative_ccc": concordance(target, prediction),
    }


def run(frame: pd.DataFrame, tasks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = frame.subject_id.unique()
    absolute_rows: list[dict[str, object]] = []
    relative_rows: list[dict[str, object]] = []
    learned_models = ("physiology_absolute", "trial_prior_plus_physiology", "subject_centered_physiology")
    total = len(tasks) * len(subjects) * len(REPRESENTATIONS) * len(learned_models)
    progress = tqdm(total=total, desc="Bias/video-prior LOSO gate", unit="fit", dynamic_ncols=True)
    for task in tasks:
        score_column = f"{task}_score"
        for held_out in subjects:
            train_mask = frame.subject_id != held_out
            test_mask = ~train_mask
            y_train = frame.loc[train_mask, score_column].to_numpy(float)
            y_test = frame.loc[test_mask, score_column].to_numpy(float)
            groups = frame.loc[train_mask, "subject_id"].to_numpy()
            train_trials = frame.loc[train_mask, "trial_id"].to_numpy(int)
            test_trials = frame.loc[test_mask, "trial_id"].to_numpy(int)
            global_prediction = np.full(len(y_test), np.mean(y_train))
            prior_prediction = trial_prior(train_trials, y_train, test_trials)
            for representation, prefix in REPRESENTATIONS.items():
                columns = [column for column in frame if column.startswith(prefix)]
                x_train = frame.loc[train_mask, columns].to_numpy(float)
                x_test = frame.loc[test_mask, columns].to_numpy(float)

                for model_name, prediction in (
                    ("global_mean", global_prediction),
                    ("trial_prior", prior_prediction),
                ):
                    absolute_rows.append(
                        {
                            "task": task,
                            "held_out_subject": held_out,
                            "representation": representation,
                            "model": model_name,
                            "alpha": np.nan,
                            **regression_metrics(y_test, prediction),
                        }
                    )

                alpha = select_alpha(x_train, y_train, groups, train_trials, "absolute")
                absolute_model = make_model(alpha).fit(x_train, y_train)
                absolute_prediction = absolute_model.predict(x_test)
                absolute_rows.append(
                    {
                        "task": task,
                        "held_out_subject": held_out,
                        "representation": representation,
                        "model": "physiology_absolute",
                        "alpha": alpha,
                        **regression_metrics(y_test, absolute_prediction),
                    }
                )
                progress.update(1)

                alpha = select_alpha(x_train, y_train, groups, train_trials, "trial_residual")
                train_prior = trial_prior(train_trials, y_train, train_trials)
                residual_model = make_model(alpha).fit(x_train, y_train - train_prior)
                residual_prediction = prior_prediction + residual_model.predict(x_test)
                absolute_rows.append(
                    {
                        "task": task,
                        "held_out_subject": held_out,
                        "representation": representation,
                        "model": "trial_prior_plus_physiology",
                        "alpha": alpha,
                        **regression_metrics(y_test, residual_prediction),
                    }
                )
                progress.update(1)

                zero_prediction = np.zeros(len(y_test))
                if representation == "stimulus_only":
                    relative_rows.append(
                        {
                            "task": task,
                            "held_out_subject": held_out,
                            "representation": "none",
                            "model": "zero_relative_change",
                            "alpha": np.nan,
                            **relative_metrics(y_test, zero_prediction),
                        }
                    )
                alpha = select_alpha(x_train, y_train, groups, train_trials, "subject_centered")
                centered_target = subject_center(y_train, groups)
                centered_model = make_model(alpha).fit(x_train, centered_target)
                centered_prediction = centered_model.predict(x_test)
                relative_rows.append(
                    {
                        "task": task,
                        "held_out_subject": held_out,
                        "representation": representation,
                        "model": "subject_centered_physiology",
                        "alpha": alpha,
                        **relative_metrics(y_test, centered_prediction),
                    }
                )
                progress.update(1)
    progress.close()
    return pd.DataFrame(absolute_rows), pd.DataFrame(relative_rows)


def paired_tests(absolute: pd.DataFrame, relative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    absolute_comparisons = (
        ("trial_prior", "global_mean"),
        ("physiology_absolute", "global_mean"),
        ("physiology_absolute", "trial_prior"),
        ("trial_prior_plus_physiology", "trial_prior"),
    )
    for (task, representation), group in absolute.groupby(["task", "representation"]):
        wide = group.pivot(index="held_out_subject", columns="model", values="mae")
        for left, right in absolute_comparisons:
            difference = wide[right] - wide[left]
            statistic, p_value = stats.wilcoxon(difference, zero_method="zsplit", mode="approx")
            rows.append(
                {
                    "task": task,
                    "representation": representation,
                    "outcome": "absolute_mae",
                    "left": left,
                    "right": right,
                    "n_subjects": len(difference),
                    "mean_improvement": float(difference.mean()),
                    "p_raw": p_value,
                    "wilcoxon_statistic": statistic,
                }
            )
    zero = relative[relative.model == "zero_relative_change"].set_index(["task", "held_out_subject"])
    for (task, representation), group in relative[relative.model == "subject_centered_physiology"].groupby(
        ["task", "representation"]
    ):
        model = group.set_index(["task", "held_out_subject"])
        difference = zero.loc[model.index, "relative_mae"] - model.relative_mae
        statistic, p_value = stats.wilcoxon(difference, zero_method="zsplit", mode="approx")
        rows.append(
            {
                "task": task,
                "representation": representation,
                "outcome": "relative_mae",
                "left": "subject_centered_physiology",
                "right": "zero_relative_change",
                "n_subjects": len(difference),
                "mean_improvement": float(difference.mean()),
                "p_raw": p_value,
                "wilcoxon_statistic": statistic,
            }
        )
    output = pd.DataFrame(rows)
    output["p_holm_within_task"] = np.nan
    for _, indices in output.groupby("task").groups.items():
        indices = np.asarray(indices)
        p = output.loc[indices, "p_raw"].to_numpy(float)
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        running = 0.0
        for rank, position in enumerate(order):
            running = max(running, (len(p) - rank) * p[position])
            adjusted[position] = min(running, 1.0)
        output.loc[indices, "p_holm_within_task"] = adjusted
    return output


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.feature_file)
    absolute, relative = run(frame, args.tasks)
    absolute.to_csv(args.output_root / "absolute_per_subject.csv", index=False)
    relative.to_csv(args.output_root / "relative_per_subject.csv", index=False)
    absolute.groupby(["task", "representation", "model"])[
        ["mae", "rmse", "spearman", "ccc", "derived_balanced_accuracy"]
    ].agg(["mean", "std", "count"]).to_csv(args.output_root / "absolute_summary.csv")
    relative.groupby(["task", "representation", "model"])[
        ["relative_mae", "relative_rmse", "relative_spearman", "relative_ccc"]
    ].agg(["mean", "std", "count"]).to_csv(args.output_root / "relative_summary.csv")
    tests = paired_tests(absolute, relative)
    tests.to_csv(args.output_root / "paired_tests.csv", index=False)
    print(tests.sort_values("p_raw").to_string(index=False))


if __name__ == "__main__":
    main()
