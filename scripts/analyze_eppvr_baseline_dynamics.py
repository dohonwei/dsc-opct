from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from tqdm.auto import tqdm


FS = 100
BASELINE_SAMPLES = 1000
WINDOW_SAMPLES = 200
STEP_SAMPLES = 100

CHANNELS = {
    "FP1": (0,),
    "FP2": (1,),
    "EDA": (2,),
    "PPG": (3,),
    "SKT": (4,),
    "EOG": (8, 9),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit baseline-anchored EPPVR response dynamics.")
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\AA发表论文的数据\dataset\EPPVR"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eppvr_baseline_dynamics"))
    return parser.parse_args()


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(1.4826 * mad, float(np.std(values)), 1e-8)
    return center, scale


def butter_filter(values: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    nyquist = FS / 2
    if low is None:
        sos = signal.butter(4, high / nyquist, btype="lowpass", output="sos")
    elif high is None:
        sos = signal.butter(4, low / nyquist, btype="highpass", output="sos")
    else:
        sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, values)


def window_feature(raw: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, raw.shape[-1] - WINDOW_SAMPLES + 1, STEP_SAMPLES)
    times = (starts + WINDOW_SAMPLES / 2 - BASELINE_SAMPLES) / FS

    if modality in {"FP1", "FP2"}:
        filtered = butter_filter(raw[0], 4.0, 45.0)
        values = np.array([np.log(np.mean(filtered[s : s + WINDOW_SAMPLES] ** 2) + 1e-12) for s in starts])
    elif modality == "EOG":
        filtered = np.stack([butter_filter(channel, 0.1, 15.0) for channel in raw])
        values = np.array(
            [np.log(np.mean(filtered[:, s : s + WINDOW_SAMPLES] ** 2) + 1e-12) for s in starts]
        )
    elif modality == "PPG":
        filtered = butter_filter(raw[0], 0.5, 5.0)
        values = np.array([np.log(np.std(filtered[s : s + WINDOW_SAMPLES]) + 1e-12) for s in starts])
    elif modality == "EDA":
        filtered = butter_filter(raw[0], None, 2.0)
        values = np.array([np.mean(filtered[s : s + WINDOW_SAMPLES]) for s in starts])
    elif modality == "SKT":
        filtered = butter_filter(raw[0], None, 0.5)
        values = np.array([np.mean(filtered[s : s + WINDOW_SAMPLES]) for s in starts])
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return times, values


def first_sustained_crossing(values: np.ndarray, threshold: float = 2.0, run: int = 3) -> float:
    above = np.abs(values) >= threshold
    hits = np.convolve(above.astype(int), np.ones(run, dtype=int), mode="valid")
    indices = np.flatnonzero(hits == run)
    return float(indices[0]) if len(indices) else np.nan


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def load_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    data = np.asarray(payload["data"], dtype=np.float64)
    labels = np.asarray(payload["label"], dtype=np.float64)
    if data.shape != (14, 10, 7000) or labels.shape[0] != 14:
        raise ValueError(f"Unexpected EPPVR shape in {path}: data={data.shape}, labels={labels.shape}")
    return data, labels


def audit_channels(subject_files: list[Path]) -> pd.DataFrame:
    rows = []
    mapping = {index: modality for modality, indices in CHANNELS.items() for index in indices}
    for subject_file in tqdm(subject_files, desc="Auditing channels", unit="subject"):
        data, _ = load_subject(subject_file)
        for channel in range(data.shape[1]):
            baseline = data[:, channel, :BASELINE_SAMPLES]
            stimulus = data[:, channel, BASELINE_SAMPLES:]
            rows.append(
                {
                    "subject_id": subject_file.stem,
                    "channel_index": channel,
                    "candidate_name": mapping.get(channel, "unused_auxiliary"),
                    "mapping_status": "candidate_requires_device_documentation",
                    "finite_fraction": float(np.isfinite(data[:, channel]).mean()),
                    "overall_min": float(np.min(data[:, channel])),
                    "overall_max": float(np.max(data[:, channel])),
                    "baseline_median": float(np.median(baseline)),
                    "baseline_std_median": float(np.median(np.std(baseline, axis=-1))),
                    "stimulus_std_median": float(np.median(np.std(stimulus, axis=-1))),
                }
            )
    return pd.DataFrame(rows)


def analyze(subject_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, curve_rows, coupling_rows = [], [], []
    for subject_file in tqdm(subject_files, desc="Extracting response dynamics", unit="subject"):
        data, labels = load_subject(subject_file)
        subject_id = subject_file.stem
        for trial in range(data.shape[0]):
            trial_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for modality, indices in CHANNELS.items():
                times, feature = window_feature(data[trial, list(indices), :], modality)
                baseline_mask = times < 0
                center, scale = robust_center_scale(feature[baseline_mask])
                z = np.clip((feature - center) / scale, -10.0, 10.0)
                stimulus_mask = times >= 0
                stim_times, stim_z = times[stimulus_mask], z[stimulus_mask]
                onset_mask = stim_times >= 2.0
                onset_times, onset_z = stim_times[onset_mask], stim_z[onset_mask]
                crossing_index = first_sustained_crossing(onset_z)
                onset = onset_times[int(crossing_index)] if np.isfinite(crossing_index) else np.nan
                peak_index = int(np.argmax(np.abs(stim_z)))
                metric_rows.append(
                    {
                        "subject_id": subject_id,
                        "trial_id": trial + 1,
                        "modality": modality,
                        "valence_score": labels[trial, 0],
                        "arousal_score": labels[trial, 1],
                        "third_score": labels[trial, 2] if labels.shape[1] > 2 else np.nan,
                        "baseline_center": center,
                        "baseline_scale": scale,
                        "onset_seconds": onset,
                        "peak_seconds": stim_times[peak_index],
                        "peak_signed_z": stim_z[peak_index],
                        "peak_abs_z": abs(stim_z[peak_index]),
                        "mean_abs_z": float(np.mean(np.abs(stim_z))),
                        "auc_abs_z": float(np.trapz(np.abs(stim_z), stim_times)),
                    }
                )
                trial_curves[modality] = (stim_times, stim_z)
                for time, value in zip(times, z, strict=True):
                    curve_rows.append(
                        {
                            "subject_id": subject_id,
                            "trial_id": trial + 1,
                            "modality": modality,
                            "time_seconds": time,
                            "signed_z": value,
                            "abs_z": abs(value),
                            "valence_score": labels[trial, 0],
                            "arousal_score": labels[trial, 1],
                        }
                    )

            eog = np.mean(data[trial, list(CHANNELS["EOG"]), :], axis=0)
            for frontal in ("FP1", "FP2"):
                fp = data[trial, CHANNELS[frontal][0], :]
                coupling_rows.append(
                    {
                        "subject_id": subject_id,
                        "trial_id": trial + 1,
                        "frontal_channel": frontal,
                        "baseline_abs_correlation": abs(stats.spearmanr(fp[:BASELINE_SAMPLES], eog[:BASELINE_SAMPLES]).statistic),
                        "stimulus_abs_correlation": abs(stats.spearmanr(fp[BASELINE_SAMPLES:], eog[BASELINE_SAMPLES:]).statistic),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(curve_rows), pd.DataFrame(coupling_rows)


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    subject_curves = (
        curves.groupby(["subject_id", "modality", "time_seconds"], as_index=False)[["signed_z", "abs_z"]].mean()
    )
    rows = []
    for (modality, time), group in subject_curves.groupby(["modality", "time_seconds"]):
        for endpoint in ("signed_z", "abs_z"):
            values = group[endpoint].to_numpy()
            rows.append(
                {
                    "modality": modality,
                    "time_seconds": time,
                    "endpoint": endpoint,
                    "mean": float(np.mean(values)),
                    "sem": float(stats.sem(values)),
                    "ci_low": float(np.mean(values) - 1.96 * stats.sem(values)),
                    "ci_high": float(np.mean(values) + 1.96 * stats.sem(values)),
                    "n_subjects": len(values),
                }
            )
    return pd.DataFrame(rows)


def paired_latency_statistics(metrics: pd.DataFrame) -> pd.DataFrame:
    subject = metrics.groupby(["subject_id", "modality"], as_index=False)["onset_seconds"].median()
    wide = subject.pivot(index="subject_id", columns="modality", values="onset_seconds")
    modalities = list(CHANNELS)
    rows = []
    raw_p = []
    for i, left in enumerate(modalities):
        for right in modalities[i + 1 :]:
            pair = wide[[left, right]].dropna()
            if len(pair) < 8:
                statistic, p_value = np.nan, np.nan
            else:
                statistic, p_value = stats.wilcoxon(pair[left], pair[right], zero_method="zsplit", mode="approx")
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n_subjects": len(pair),
                    "median_left_seconds": float(pair[left].median()) if len(pair) else np.nan,
                    "median_right_seconds": float(pair[right].median()) if len(pair) else np.nan,
                    "median_difference_seconds": float((pair[left] - pair[right]).median()) if len(pair) else np.nan,
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_value,
                }
            )
            raw_p.append(p_value)
    finite = [p for p in raw_p if np.isfinite(p)]
    adjusted = iter(holm_adjust(finite))
    for row in rows:
        row["p_holm"] = next(adjusted) if np.isfinite(row["p_raw"]) else np.nan
    return pd.DataFrame(rows)


def label_association_statistics(curves: pd.DataFrame) -> pd.DataFrame:
    stimulus = curves[curves.time_seconds >= 2.0].copy()
    stimulus["phase"] = pd.cut(
        stimulus.time_seconds,
        bins=[2.0, 10.0, 30.0, 60.0],
        labels=["early_2_10s", "middle_10_30s", "late_30_60s"],
        include_lowest=True,
    )
    rows = []
    for label_name, score_column in (("arousal", "arousal_score"), ("valence", "valence_score")):
        for endpoint in ("signed_z", "abs_z"):
            trial_phase = (
                stimulus.groupby(
                    ["subject_id", "trial_id", "modality", "phase", score_column],
                    observed=True,
                    as_index=False,
                )[endpoint]
                .mean()
            )
            trial_phase["label"] = np.where(trial_phase[score_column] >= 5, "high", "low")
            subject_label = (
                trial_phase.groupby(["subject_id", "modality", "phase", "label"], observed=True, as_index=False)[endpoint]
                .mean()
            )
            for (modality, phase), group in subject_label.groupby(["modality", "phase"], observed=True):
                wide = group.pivot(index="subject_id", columns="label", values=endpoint).dropna()
                if len(wide) < 8 or not {"high", "low"}.issubset(wide.columns):
                    statistic, p_value = np.nan, np.nan
                    differences = np.array([], dtype=float)
                else:
                    differences = (wide.high - wide.low).to_numpy()
                    statistic, p_value = stats.wilcoxon(differences, zero_method="zsplit", mode="approx")
                rows.append(
                    {
                        "label_dimension": label_name,
                        "endpoint": endpoint,
                        "modality": modality,
                        "phase": str(phase),
                        "n_subjects": len(wide),
                        "mean_high_minus_low": float(np.mean(differences)) if len(differences) else np.nan,
                        "median_high_minus_low": float(np.median(differences)) if len(differences) else np.nan,
                        "cohens_dz": float(np.mean(differences) / np.std(differences, ddof=1))
                        if len(differences) > 1 and np.std(differences, ddof=1) > 0
                        else np.nan,
                        "wilcoxon_statistic": statistic,
                        "p_raw": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    for (label_name, endpoint), indices in result.groupby(["label_dimension", "endpoint"]).groups.items():
        finite_indices = [index for index in indices if np.isfinite(result.loc[index, "p_raw"])]
        adjusted = holm_adjust(result.loc[finite_indices, "p_raw"].tolist())
        result.loc[finite_indices, "p_holm"] = adjusted
    return result


def plot_curves(summary: pd.DataFrame, output: Path) -> None:
    colors = {"FP1": "#2F6B9A", "FP2": "#5B8DB8", "EOG": "#9A4D63", "EDA": "#26856A", "PPG": "#D17B35", "SKT": "#665191"}
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    for axis, endpoint, title in zip(axes, ("signed_z", "abs_z"), ("Signed change from baseline", "Response magnitude from baseline"), strict=True):
        for modality in CHANNELS:
            part = summary[(summary.modality == modality) & (summary.endpoint == endpoint)].sort_values("time_seconds")
            axis.plot(part.time_seconds, part["mean"], label=modality, color=colors[modality], linewidth=1.8)
            axis.fill_between(part.time_seconds, part.ci_low, part.ci_high, color=colors[modality], alpha=0.13)
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_ylabel("Baseline-referenced z")
        axis.set_title(title, loc="left", fontsize=11)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[-1].set_xlabel("Time relative to stimulus onset (s)")
    for axis in axes:
        axis.axvline(0, color="#222222", linestyle="--", linewidth=1.0)
    axes[0].legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latency(metrics: pd.DataFrame, output: Path) -> None:
    subject = metrics.groupby(["subject_id", "modality"], as_index=False)["onset_seconds"].median()
    arrays = [subject.loc[subject.modality == modality, "onset_seconds"].dropna().to_numpy() for modality in CHANNELS]
    fig, axis = plt.subplots(figsize=(8.5, 4.4))
    parts = axis.violinplot(arrays, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], ("#2F6B9A", "#5B8DB8", "#26856A", "#D17B35", "#665191", "#9A4D63"), strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.65)
    axis.set_xticks(np.arange(1, len(CHANNELS) + 1), list(CHANNELS))
    axis.set_ylabel("Median response onset (s)")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_label_differences(label_stats: pd.DataFrame, output: Path) -> None:
    colors = {"FP1": "#2F6B9A", "FP2": "#5B8DB8", "EOG": "#9A4D63", "EDA": "#26856A", "PPG": "#D17B35", "SKT": "#665191"}
    phases = ["early_2_10s", "middle_10_30s", "late_30_60s"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for axis, label_name in zip(axes, ("arousal", "valence"), strict=True):
        part = label_stats[(label_stats.label_dimension == label_name) & (label_stats.endpoint == "abs_z")]
        for modality in CHANNELS:
            values = (
                part[part.modality == modality]
                .set_index("phase")
                .reindex(phases)["mean_high_minus_low"]
                .to_numpy()
            )
            axis.plot(np.arange(3), values, marker="o", color=colors[modality], label=modality, linewidth=1.6)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xticks(np.arange(3), ["2-10", "10-30", "30-60"])
        axis.set_xlabel("Stimulus phase (s)")
        axis.set_title(label_name.capitalize(), loc="left")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("High - low response magnitude (z)")
    axes[1].legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    subject_files = sorted(args.data_root.glob("s*.dat"))
    if len(subject_files) != 30:
        raise ValueError(f"Expected 30 EPPVR subject files, found {len(subject_files)} in {args.data_root}")

    audit = audit_channels(subject_files)
    metrics, curves, coupling = analyze(subject_files)
    summary = summarize_curves(curves)
    latency_stats = paired_latency_statistics(metrics)
    label_stats = label_association_statistics(curves)

    audit.to_csv(args.output_root / "channel_audit.csv", index=False)
    metrics.to_csv(args.output_root / "trial_response_metrics.csv", index=False)
    curves.to_csv(args.output_root / "trial_response_curves.csv", index=False)
    summary.to_csv(args.output_root / "group_response_curves.csv", index=False)
    latency_stats.to_csv(args.output_root / "paired_latency_statistics.csv", index=False)
    label_stats.to_csv(args.output_root / "label_association_statistics.csv", index=False)
    coupling.to_csv(args.output_root / "eog_frontal_coupling.csv", index=False)
    plot_curves(summary, args.output_root / "baseline_anchored_response_curves.png")
    plot_latency(metrics, args.output_root / "response_onset_distributions.png")
    plot_label_differences(label_stats, args.output_root / "label_response_differences.png")

    print(f"Analyzed {len(subject_files)} subjects, {metrics.trial_id.count() // len(CHANNELS)} trials")
    print(metrics.groupby("modality")["onset_seconds"].agg(["count", "median", "mean"]).round(3))
    significant = label_stats[label_stats.p_holm < 0.05]
    print(f"Holm-significant label-response rows: {len(significant)}/{len(label_stats)}")
    print(f"Outputs: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
