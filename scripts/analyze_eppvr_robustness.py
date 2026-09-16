from __future__ import annotations

import argparse
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from tqdm.auto import tqdm


FS = 100
BASELINE_SAMPLES = 1000
CHANNEL_LABELS = {
    0: "FP1_confirmed_by_legacy_code",
    1: "FP2_confirmed_by_legacy_code",
    2: "EDA_confirmed_by_legacy_code",
    3: "PPG_confirmed_by_legacy_code",
    4: "SKT_confirmed_by_legacy_code",
    5: "auxiliary_unknown",
    6: "auxiliary_unknown",
    7: "auxiliary_unknown",
    8: "EOG_candidate",
    9: "EOG_candidate",
}
MODALITIES = ("FP1", "FP2", "EDA", "PPG", "SKT", "EOG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness checks for EPPVR baseline dynamics.")
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\AA发表论文的数据\dataset\EPPVR"))
    parser.add_argument("--analysis-root", type=Path, default=Path("outputs/eppvr_baseline_dynamics"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eppvr_baseline_robustness"))
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def load_subject(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    return np.asarray(payload["data"], dtype=np.float64)


def spectral_audit(subject_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectral_rows: list[dict[str, float | int | str]] = []
    correlation_rows: list[dict[str, float | int | str]] = []
    bands = {"slow_0p1_1": (0.1, 1), "physio_1_15": (1, 15), "high_15_45": (15, 45)}

    for subject_file in tqdm(subject_files, desc="Auditing channel identity", unit="subject"):
        data = load_subject(subject_file)
        subject_id = subject_file.stem
        for segment_name, segment in (
            ("baseline", data[:, :, :BASELINE_SAMPLES]),
            ("stimulus", data[:, :, BASELINE_SAMPLES:]),
        ):
            flattened = segment.transpose(1, 0, 2).reshape(10, -1)
            corr = stats.spearmanr(flattened, axis=1).statistic
            for left, right in combinations(range(10), 2):
                correlation_rows.append(
                    {
                        "subject_id": subject_id,
                        "segment": segment_name,
                        "channel_left": left,
                        "channel_right": right,
                        "abs_spearman": abs(float(corr[left, right])),
                    }
                )

            for channel in range(10):
                frequencies, psd = signal.welch(flattened[channel], fs=FS, nperseg=512, noverlap=256)
                total_mask = (frequencies >= 0.1) & (frequencies <= 45)
                total_power = np.trapz(psd[total_mask], frequencies[total_mask])
                cumulative = np.cumsum(psd[total_mask])
                edge_index = min(np.searchsorted(cumulative, 0.95 * cumulative[-1]), cumulative.size - 1)
                row: dict[str, float | int | str] = {
                    "subject_id": subject_id,
                    "segment": segment_name,
                    "channel_index": channel,
                    "candidate_name": CHANNEL_LABELS[channel],
                    "unique_fraction": float(np.unique(flattened[channel]).size / flattened[channel].size),
                    "integer_like_fraction": float(
                        np.mean(np.isclose(flattened[channel], np.round(flattened[channel]), atol=1e-8))
                    ),
                    "spectral_edge_95_hz": float(frequencies[total_mask][edge_index]),
                }
                for name, (low, high) in bands.items():
                    mask = (frequencies >= low) & (frequencies < high)
                    band_power = np.trapz(psd[mask], frequencies[mask])
                    row[f"relative_power_{name}"] = float(band_power / max(total_power, 1e-30))
                spectral_rows.append(row)

    return pd.DataFrame(spectral_rows), pd.DataFrame(correlation_rows)


def first_sustained_onset(
    times: np.ndarray, values: np.ndarray, threshold: float, run: int, exclusion: float
) -> float:
    eligible = times >= exclusion
    times = times[eligible]
    values = values[eligible]
    hits = np.convolve((np.abs(values) >= threshold).astype(int), np.ones(run, dtype=int), mode="valid")
    indices = np.flatnonzero(hits == run)
    return float(times[indices[0]]) if len(indices) else np.nan


def onset_sensitivity(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stimulus = curves[curves.time_seconds >= 0].copy()
    settings = [
        (threshold, run, exclusion)
        for threshold in (1.5, 2.0, 2.5, 3.0)
        for run in (2, 3, 5)
        for exclusion in (1.0, 2.0, 3.0, 5.0)
    ]
    groups = list(stimulus.groupby(["subject_id", "trial_id", "modality"], sort=False))
    onset_rows: list[dict[str, float | int | str]] = []
    for threshold, run, exclusion in tqdm(settings, desc="Testing onset definitions", unit="setting"):
        for (subject_id, trial_id, modality), group in groups:
            onset_rows.append(
                {
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "modality": modality,
                    "threshold_z": threshold,
                    "sustained_windows": run,
                    "transition_exclusion_seconds": exclusion,
                    "onset_seconds": first_sustained_onset(
                        group.time_seconds.to_numpy(), group.signed_z.to_numpy(), threshold, run, exclusion
                    ),
                }
            )
    onsets = pd.DataFrame(onset_rows)
    summaries = []
    for keys, group in onsets.groupby(
        ["threshold_z", "sustained_windows", "transition_exclusion_seconds", "modality"]
    ):
        threshold, run, exclusion, modality = keys
        subject = group.groupby("subject_id").onset_seconds.median().dropna()
        summaries.append(
            {
                "threshold_z": threshold,
                "sustained_windows": run,
                "transition_exclusion_seconds": exclusion,
                "modality": modality,
                "detected_trials": int(group.onset_seconds.notna().sum()),
                "detected_fraction": float(group.onset_seconds.notna().mean()),
                "subjects_with_detection": int(len(subject)),
                "subject_median_onset_seconds": float(subject.median()) if len(subject) else np.nan,
            }
        )
    return onsets, pd.DataFrame(summaries)


def bootstrap_pairwise(onsets: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    setting_columns = ["threshold_z", "sustained_windows", "transition_exclusion_seconds"]
    grouped = list(onsets.groupby(setting_columns))
    for setting, group in tqdm(grouped, desc="Bootstrapping modality contrasts", unit="setting"):
        subject = group.groupby(["subject_id", "modality"]).onset_seconds.median().unstack()
        for left, right in combinations(MODALITIES, 2):
            pair = subject[[left, right]].dropna()
            differences = (pair[left] - pair[right]).to_numpy()
            if len(differences) < 8:
                low = high = np.nan
            else:
                samples = rng.choice(differences, size=(iterations, len(differences)), replace=True)
                medians = np.median(samples, axis=1)
                low, high = np.quantile(medians, [0.025, 0.975])
            rows.append(
                {
                    "threshold_z": setting[0],
                    "sustained_windows": setting[1],
                    "transition_exclusion_seconds": setting[2],
                    "left": left,
                    "right": right,
                    "n_subjects": len(differences),
                    "median_difference_seconds": float(np.median(differences)) if len(differences) else np.nan,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "direction_stable": bool(low > 0 or high < 0) if np.isfinite(low) else False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    subject_files = sorted(args.data_root.glob("s*.dat"))
    if not subject_files:
        raise FileNotFoundError(f"No EPPVR subject files found in {args.data_root}")
    curves_path = args.analysis_root / "trial_response_curves.csv"
    if not curves_path.exists():
        raise FileNotFoundError(f"Run baseline dynamics analysis first: missing {curves_path}")

    spectra, correlations = spectral_audit(subject_files)
    curves = pd.read_csv(curves_path)
    onsets, sensitivity = onset_sensitivity(curves)
    pairwise = bootstrap_pairwise(onsets, args.bootstrap_iterations, args.seed)

    spectra.to_csv(args.output_root / "channel_spectral_audit.csv", index=False)
    correlations.to_csv(args.output_root / "channel_pair_correlations.csv", index=False)
    sensitivity.to_csv(args.output_root / "onset_sensitivity_summary.csv", index=False)
    pairwise.to_csv(args.output_root / "onset_pairwise_bootstrap.csv", index=False)

    stability = pairwise.groupby(["left", "right"], as_index=False).agg(
        tested_settings=("direction_stable", "size"),
        stable_settings=("direction_stable", "sum"),
        median_difference_seconds=("median_difference_seconds", "median"),
    )
    stability["stable_fraction"] = stability.stable_settings / stability.tested_settings
    stability.to_csv(args.output_root / "onset_contrast_stability.csv", index=False)
    print(f"Saved robustness outputs to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
