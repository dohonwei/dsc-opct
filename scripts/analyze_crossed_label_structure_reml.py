from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from tqdm.auto import tqdm

from stimulus_identity_contract import CROSSED_TRIAL_DATASETS, require_crossed_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REML crossed subject-by-stimulus label audit.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(CROSSED_TRIAL_DATASETS),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/stimulus_shortcut_audit_v2_crossed_valid"),
    )
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def fit_crossed_reml(frame: pd.DataFrame, score_column: str) -> dict[str, float | bool | str]:
    data = frame[["subject_id", "trial_id", score_column]].rename(columns={score_column: "score"}).copy()
    data["subject_id"] = data.subject_id.astype(str)
    data["trial_id"] = data.trial_id.astype(str)
    model = sm.MixedLM.from_formula(
        "score ~ 1",
        groups=np.ones(len(data)),
        re_formula="0",
        vc_formula={"stimulus": "0 + C(trial_id)", "subject": "0 + C(subject_id)"},
        data=data,
    )
    result = model.fit(reml=True, method="lbfgs", maxiter=2000, disp=False)
    components = dict(zip(result.model.exog_vc.names, result.vcomp, strict=True))
    subject = float(components["subject"])
    stimulus = float(components["stimulus"])
    residual = float(result.scale)
    total = subject + stimulus + residual
    return {
        "subject_variance": subject,
        "stimulus_variance": stimulus,
        "residual_variance": residual,
        "subject_variance_fraction": subject / total,
        "stimulus_variance_fraction": stimulus / total,
        "residual_variance_fraction": residual / total,
        "reml_converged": bool(result.converged),
        "reml_optimizer_message": str(getattr(result, "message", "")),
    }


def permutation_test(
    identifiers: np.ndarray, labels: np.ndarray, permutations: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    observed = normalized_mutual_info_score(identifiers, labels, average_method="arithmetic")
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        null[index] = normalized_mutual_info_score(
            identifiers, rng.permutation(labels), average_method="arithmetic"
        )
    p_value = (1 + np.sum(null >= observed)) / (permutations + 1)
    return float(observed), float(null.mean()), float(p_value)


def main() -> None:
    args = parse_args()
    require_crossed_datasets(args.datasets, context="Crossed REML label audit")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    specifications = [item.split("=", 1) for item in args.datasets]
    progress = tqdm(total=len(specifications) * 2, desc="Crossed REML audit", unit="task", dynamic_ncols=True)
    for dataset, path in specifications:
        frame = pd.read_csv(path)
        if frame.duplicated(["subject_id", "trial_id"]).any():
            raise ValueError(f"{dataset} contains duplicate subject/stimulus rows")
        for task in ("arousal", "valence"):
            binary = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
            nmi, null_nmi, p_nmi = permutation_test(
                frame.trial_id.to_numpy(), binary, args.permutations, rng
            )
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "n_subjects": frame.subject_id.nunique(),
                    "n_stimuli": frame.trial_id.nunique(),
                    "n_observed_pairs": len(frame),
                    "crossed_completeness": len(frame)
                    / (frame.subject_id.nunique() * frame.trial_id.nunique()),
                    **fit_crossed_reml(frame, f"{task}_score"),
                    "stimulus_binary_nmi": nmi,
                    "stimulus_binary_nmi_permutation_mean": null_nmi,
                    "stimulus_binary_nmi_permutation_p": p_nmi,
                    "stimulus_binary_ami": adjusted_mutual_info_score(frame.trial_id, binary),
                    "subject_binary_ami": adjusted_mutual_info_score(frame.subject_id, binary),
                }
            )
            progress.update(1)
    progress.close()
    output = pd.DataFrame(rows)
    output.to_csv(args.output_root / "crossed_label_variance_reml.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
