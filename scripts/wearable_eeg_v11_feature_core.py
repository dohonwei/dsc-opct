from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.signal import detrend, welch


WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0
WELCH_SECONDS = 2.0
MINIMUM_VALID_WINDOWS = 3


def trajectory(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    time = np.linspace(-1.0, 1.0, values.shape[0])
    centered = time - time.mean()
    denominator = float(np.sum(centered**2))
    slope = float(centered @ values / max(denominator, 1e-12))
    return float(values.mean()), float(values.std()), slope


def add_trajectory(row: dict[str, object], prefix: str, values: np.ndarray) -> None:
    mean, standard_deviation, slope = trajectory(values)
    row[f"stimulus__{prefix}_mean"] = mean
    row[f"stimulus__{prefix}_std"] = standard_deviation
    row[f"stimulus__{prefix}_slope"] = slope


def window_band_features(
    data_uv: np.ndarray,
    sampling_rate: float,
    channels: Sequence[str],
    bands_hz: Mapping[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    data_uv = np.asarray(data_uv, dtype=float)
    if data_uv.ndim != 2 or data_uv.shape[0] != len(channels):
        raise ValueError(
            f"Expected channels by samples with {len(channels)} channels, "
            f"found {data_uv.shape}"
        )
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("Registered channel names must be nonempty and unique")
    if not bands_hz:
        raise ValueError("At least one frequency band is required")
    maximum_frequency = max(high for _, high in bands_hz.values())
    if not np.isfinite(sampling_rate) or sampling_rate < 2 * maximum_frequency:
        raise ValueError(f"Sampling rate cannot support the frozen bands: {sampling_rate}")
    window_samples = int(round(WINDOW_SECONDS * sampling_rate))
    step_samples = int(round(STEP_SECONDS * sampling_rate))
    if data_uv.shape[1] < window_samples:
        raise ValueError("Video segment is shorter than one registered analysis window")
    starts = range(0, data_uv.shape[1] - window_samples + 1, step_samples)
    absolute_rows = []
    for start in starts:
        window = data_uv[:, start : start + window_samples]
        if not np.isfinite(window).all():
            continue
        window = detrend(window, axis=1, type="linear")
        if np.any(np.var(window, axis=1) <= 1e-12):
            continue
        nperseg = min(window_samples, int(round(WELCH_SECONDS * sampling_rate)))
        frequencies, density = welch(
            window,
            fs=sampling_rate,
            axis=1,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend=False,
            scaling="density",
        )
        band_values = []
        for low, high in bands_hz.values():
            selected = (frequencies >= low) & (frequencies <= high)
            if selected.sum() < 2:
                raise ValueError(
                    f"Insufficient Welch bins for registered band {low}-{high} Hz"
                )
            band_values.append(
                np.trapz(density[:, selected], frequencies[selected], axis=1)
            )
        absolute_rows.append(np.stack(band_values, axis=1))
    if len(absolute_rows) < MINIMUM_VALID_WINDOWS:
        raise ValueError(
            f"Only {len(absolute_rows)} valid windows; minimum is {MINIMUM_VALID_WINDOWS}"
        )
    absolute = np.stack(absolute_rows, axis=0)
    relative = absolute / np.maximum(absolute.sum(axis=2, keepdims=True), 1e-20)
    return absolute, relative


def trial_features(
    data_uv: np.ndarray,
    sampling_rate: float,
    subject_id: str,
    trial_id: str | int,
    channels: Sequence[str],
    bilateral_pairs: Sequence[tuple[str, str]],
    bands_hz: Mapping[str, tuple[float, float]],
) -> dict[str, object]:
    data_uv = np.asarray(data_uv, dtype=float)
    data_uv = data_uv - data_uv.mean(axis=0, keepdims=True)
    absolute, relative = window_band_features(
        data_uv, sampling_rate, channels, bands_hz
    )
    log_power = np.log10(np.maximum(absolute, 1e-20))
    row: dict[str, object] = {
        "subject_id": str(subject_id),
        "trial_id": trial_id,
        "sampling_rate_hz": float(sampling_rate),
        "n_windows": int(absolute.shape[0]),
    }
    channel_lookup = {name: index for index, name in enumerate(channels)}
    for left, right in bilateral_pairs:
        if left not in channel_lookup or right not in channel_lookup or left == right:
            raise ValueError(f"Invalid registered bilateral pair: {(left, right)}")
    for band_index, band in enumerate(bands_hz):
        for channel_index, channel in enumerate(channels):
            add_trajectory(
                row,
                f"log_power_{channel}_{band}",
                log_power[:, channel_index, band_index],
            )
            add_trajectory(
                row,
                f"relative_power_{channel}_{band}",
                relative[:, channel_index, band_index],
            )
        for left, right in bilateral_pairs:
            left_index = channel_lookup[left]
            right_index = channel_lookup[right]
            pair = f"{left}_{right}"
            add_trajectory(
                row,
                f"log_difference_{pair}_{band}",
                log_power[:, left_index, band_index]
                - log_power[:, right_index, band_index],
            )
            denominator = np.maximum(
                np.abs(relative[:, left_index, band_index])
                + np.abs(relative[:, right_index, band_index]),
                1e-20,
            )
            add_trajectory(
                row,
                f"normalized_asymmetry_{pair}_{band}",
                (
                    relative[:, left_index, band_index]
                    - relative[:, right_index, band_index]
                )
                / denominator,
            )
    feature_values = np.asarray(
        [value for key, value in row.items() if key.startswith("stimulus__")],
        dtype=float,
    )
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite wearable-EEG trial-level feature")
    return row


def representation_columns(row: Mapping[str, object]) -> dict[str, list[str]]:
    feature_columns = sorted(key for key in row if key.startswith("stimulus__"))
    return {
        "all": feature_columns,
        "relative_power": [
            column
            for column in feature_columns
            if column.startswith("stimulus__relative_power_")
        ],
        "normalized_asymmetry": [
            column
            for column in feature_columns
            if column.startswith("stimulus__normalized_asymmetry_")
        ],
    }
