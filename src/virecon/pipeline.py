from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .data import FeatureScaler, ReconstructionData, source_tensor
from .models import MLPReconstructor, RegionalUncertaintyReconstructor


@dataclass(frozen=True)
class ModelConfig:
    hidden: int
    dropout: float
    batch_size: int
    epochs: int
    patience: int
    lr: float
    l1_weight: float
    nll_weight: float
    correlation_weight: float
    mc_samples: int = 30
    validation_subjects: int = 3


@dataclass(frozen=True)
class ExperimentResults:
    reconstruction: pd.DataFrame
    feature_groups: pd.DataFrame
    downstream: pd.DataFrame


FEATURE_GROUP_PREFIXES = {
    "energy_pw": ("energy_", "pw_"),
    "psd": ("psd_",),
    "de": ("de_",),
    "asi_evi": ("asi_", "evi_"),
    "dasm_rasm_dcau": ("dasm_", "rasm_", "dcau_"),
}

REGIONAL_METHODS = {"regional", "regional_no_geo", "regional_no_gate", "regional_no_uncertainty"}
NEURAL_METHODS = {"mlp", "mc_dropout"} | REGIONAL_METHODS


def _correlation(true: np.ndarray, pred: np.ndarray) -> float:
    values = [
        np.corrcoef(true[:, index], pred[:, index])[0, 1]
        for index in range(true.shape[1])
        if np.std(true[:, index]) > 0 and np.std(pred[:, index]) > 0
    ]
    return float(np.nanmean(values)) if values else float("nan")


def _metrics(true: np.ndarray, pred: np.ndarray, gaussian_nll: float | None = None) -> dict[str, float]:
    true_log, pred_log = FeatureScaler._stabilize(true), FeatureScaler._stabilize(pred)
    residual = true_log - pred_log
    denominator = np.sum((true_log - true_log.mean(axis=0, keepdims=True)) ** 2)
    result = {
        "log_mae": float(np.mean(np.abs(residual))),
        "log_rmse": float(np.sqrt(np.mean(residual**2))),
        "log_r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan"),
        "log_feature_correlation": _correlation(true_log, pred_log),
    }
    if gaussian_nll is not None:
        result["gaussian_nll"] = gaussian_nll
    return result


def _uncertainty_metrics(true: np.ndarray, mean: np.ndarray, log_variance: np.ndarray) -> dict[str, float]:
    standard_deviation = np.exp(0.5 * log_variance)
    absolute_error = np.abs(true - mean)
    flat_error, flat_std = absolute_error.ravel(), standard_deviation.ravel()
    correlation = (
        float(np.corrcoef(flat_error, flat_std)[0, 1])
        if np.std(flat_error) > 0 and np.std(flat_std) > 0
        else float("nan")
    )
    return {
        "coverage_90": float(np.mean(absolute_error <= 1.644854 * standard_deviation)),
        "coverage_95": float(np.mean(absolute_error <= 1.959964 * standard_deviation)),
        "mean_predictive_std": float(np.mean(standard_deviation)),
        "error_uncertainty_correlation": correlation,
    }


def _composite_score(true: np.ndarray, pred: np.ndarray, correlation_weight: float) -> float:
    return float(np.mean(np.abs(true - pred)) + correlation_weight * (1.0 - _correlation(true, pred)))


def _select_residual_scale(
    true: np.ndarray, base: np.ndarray, residual: np.ndarray, correlation_weight: float
) -> float:
    candidates = np.linspace(0.0, 1.0, 9)
    scores = [_composite_score(true, base + scale * residual, correlation_weight) for scale in candidates]
    return float(candidates[int(np.nanargmin(scores))])


def _select_base_blend(
    true: np.ndarray, interpolation: np.ndarray, mlp: np.ndarray, correlation_weight: float
) -> float:
    """Select the MLP share using only the inner validation subject."""
    candidates = np.linspace(0.0, 1.0, 5)
    scores = [
        _composite_score(true, (1.0 - weight) * interpolation + weight * mlp, correlation_weight)
        for weight in candidates
    ]
    return float(candidates[int(np.nanargmin(scores))])


def _feature_group_rows(
    data: ReconstructionData, true: np.ndarray, pred: np.ndarray, metadata: dict[str, object]
) -> list[dict[str, object]]:
    names = [name.lower() for name in data.feature_names]
    rows = []
    for group, prefixes in FEATURE_GROUP_PREFIXES.items():
        mask = np.asarray([name.startswith(prefixes) for name in names])
        rows.append({**metadata, "feature_group": group, **_metrics(true[:, mask], pred[:, mask])})
    return rows


def _train_neural(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    config: ModelConfig,
    device: str,
    seed: int,
    train_base: np.ndarray | None = None,
    val_base: np.ndarray | None = None,
):
    torch.manual_seed(seed)
    if method == "mlp":
        model: nn.Module = MLPReconstructor(train_x.shape[1], hidden=config.hidden, dropout=config.dropout)
    else:
        model = RegionalUncertaintyReconstructor(train_x.shape[1], hidden=config.hidden, dropout=config.dropout)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    train_base = np.zeros_like(train_y) if train_base is None else train_base.astype(np.float32)
    val_base = np.zeros_like(val_y) if val_base is None else val_base.astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_base)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    vx, vy = torch.from_numpy(val_x).to(device), torch.from_numpy(val_y).to(device)
    vb = torch.from_numpy(val_base).to(device)
    model.eval()
    with torch.no_grad():
        initial_output = model(vx)
        initial_mean = (initial_output[0] if method in REGIONAL_METHODS else initial_output) + vb
        initial_centered = initial_mean - initial_mean.mean(dim=0, keepdim=True)
        target_centered = vy - vy.mean(dim=0, keepdim=True)
        initial_corr = nn.functional.cosine_similarity(initial_centered.T, target_centered.T, dim=1).mean()
        best = (nn.functional.l1_loss(initial_mean, vy) + config.correlation_weight * (1.0 - initial_corr)).item()
    stale = 0
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for _ in range(config.epochs):
        model.train()
        for x, y, base in loader:
            x, y, base = x.to(device), y.to(device), base.to(device)
            output = model(x)
            if method in REGIONAL_METHODS:
                residual, logvar = output
                mean = base + residual
                nll = 0.5 * (logvar + (y - mean).pow(2) * torch.exp(-logvar)).mean()
                centered_mean = mean - mean.mean(dim=0, keepdim=True)
                centered_y = y - y.mean(dim=0, keepdim=True)
                correlation = nn.functional.cosine_similarity(centered_mean.T, centered_y.T, dim=1).mean()
                loss = (
                    config.l1_weight * nn.functional.l1_loss(mean, y)
                    + config.nll_weight * nll
                    + config.correlation_weight * (1.0 - correlation)
                )
            else:
                mean = output
                loss = nn.functional.l1_loss(mean, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(vx)
            mean = (output[0] if method in REGIONAL_METHODS else output) + vb
            centered_mean = mean - mean.mean(dim=0, keepdim=True)
            centered_y = vy - vy.mean(dim=0, keepdim=True)
            corr = nn.functional.cosine_similarity(centered_mean.T, centered_y.T, dim=1).mean()
            score = (nn.functional.l1_loss(mean, vy) + config.correlation_weight * (1.0 - corr)).item()
        if score < best:
            best, stale = score, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if state is None:
        raise RuntimeError(f"{method} training did not produce a valid checkpoint.")
    model.load_state_dict(state)
    return model.cpu()


def _predict_neural(model: nn.Module, method: str, values: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray | None]:
    model = model.to(device).eval()
    means, logvars = [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=2048)
    with torch.no_grad():
        for (batch,) in loader:
            output = model(batch.to(device))
            if method in REGIONAL_METHODS:
                mean, logvar = output
                logvars.append(logvar.cpu().numpy())
            else:
                mean = output
            means.append(mean.cpu().numpy())
    return np.concatenate(means), np.concatenate(logvars) if logvars else None


def _predict_mc_dropout(
    model: nn.Module, values: np.ndarray, device: str, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device).eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    means, logvars = [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=2048)
    with torch.no_grad():
        for (batch,) in loader:
            draws = torch.stack([model(batch.to(device)) for _ in range(samples)], dim=0)
            means.append(draws.mean(dim=0).cpu().numpy())
            logvars.append(torch.log(draws.var(dim=0, unbiased=False).clamp_min(1e-6)).cpu().numpy())
    return np.concatenate(means), np.concatenate(logvars)


REGION_DISTANCE_TO_FP12 = {
    # Mean homologous-electrode Euclidean distances to Fp1/Fp2 from MNE's
    # standard_1020 montage (metres); no outcome data enter these weights.
    "af34": 0.029937752069520582,
    "f34": 0.061288699891959106,
    "fc12": 0.10430133965652977,
    "t78": 0.11407099879848837,
}


def _geometry_interpolation(values: np.ndarray, regions: list[str]) -> np.ndarray:
    """Inverse-distance interpolation in the paired paper40 feature space."""
    distances = np.asarray([REGION_DISTANCE_TO_FP12[name] for name in regions], dtype=np.float64)
    weights = distances**-2
    weights /= weights.sum()
    return np.sum(values * weights[None, :, None], axis=1).astype(np.float32)


def _trial_pool(values: np.ndarray, subjects: np.ndarray, trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.stack([subjects, trials], axis=1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    pooled = np.zeros((len(unique_keys), values.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(pooled, inverse, values)
    return pooled / counts[:, None], unique_keys


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    support = matrix.sum(axis=1)
    recalls = np.divide(np.diag(matrix), support, out=np.full(2, np.nan), where=support > 0)
    return {
        "balanced_accuracy": float(np.nanmean(recalls)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size == 2 else float("nan"),
    }


def _downstream_rows(
    data: ReconstructionData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    source_train: np.ndarray,
    source_test: np.ndarray,
    reconstructed_train: np.ndarray,
    reconstructed_test: np.ndarray,
    metadata: dict[str, object],
    seed: int,
) -> list[dict[str, object]]:
    source_train = FeatureScaler._stabilize(source_train.reshape(len(source_train), -1))
    source_test = FeatureScaler._stabilize(source_test.reshape(len(source_test), -1))
    real_train, real_test = data.target[train_indices], data.target[test_indices]
    representations = {
        "source_only": (source_train, source_test),
        "reconstructed_fp12": (FeatureScaler._stabilize(reconstructed_train), FeatureScaler._stabilize(reconstructed_test)),
        "source_plus_reconstructed": (
            np.concatenate([source_train, FeatureScaler._stabilize(reconstructed_train)], axis=1),
            np.concatenate([source_test, FeatureScaler._stabilize(reconstructed_test)], axis=1),
        ),
        "oracle_real_fp12": (FeatureScaler._stabilize(real_train), FeatureScaler._stabilize(real_test)),
    }
    rows = []
    train_subjects, train_trials = data.subjects[train_indices], data.trials[train_indices]
    test_subjects, test_trials = data.subjects[test_indices], data.trials[test_indices]
    for target_name, labels in data.labels.items():
        train_labels, train_keys = _trial_pool(labels[train_indices, None], train_subjects, train_trials)
        test_labels, test_keys = _trial_pool(labels[test_indices, None], test_subjects, test_trials)
        y_train, y_test = np.rint(train_labels[:, 0]).astype(int), np.rint(test_labels[:, 0]).astype(int)
        for representation, (x_train, x_test) in representations.items():
            x_train, x_train_keys = _trial_pool(x_train, train_subjects, train_trials)
            x_test, x_test_keys = _trial_pool(x_test, test_subjects, test_trials)
            if not (np.array_equal(train_keys, x_train_keys) and np.array_equal(test_keys, x_test_keys)):
                raise RuntimeError("Trial aggregation keys became misaligned.")
            scaler = StandardScaler().fit(x_train)
            classifier = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=seed, solver="liblinear"
            ).fit(scaler.transform(x_train), y_train)
            score = classifier.predict_proba(scaler.transform(x_test))[:, 1]
            rows.append(
                {
                    **metadata,
                    "target": target_name,
                    "representation": representation,
                    "n_test_trials": len(y_test),
                    **_binary_metrics(y_test, score),
                }
            )
    return rows


def _save_fold_predictions(
    root: Path,
    data: ReconstructionData,
    held: int,
    budget: str,
    method: str,
    test_indices: np.ndarray,
    predicted: np.ndarray,
    true_standardized: np.ndarray,
    predicted_standardized: np.ndarray,
    log_variance: np.ndarray | None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject_id": data.subjects[test_indices],
        "trial_id": data.trials[test_indices],
        "true_fp12": data.target[test_indices].astype(np.float32),
        "predicted_fp12": predicted.astype(np.float32),
        "true_standardized": true_standardized.astype(np.float32),
        "predicted_standardized": predicted_standardized.astype(np.float32),
        "labels_arousal": data.labels["arousal"][test_indices],
        "labels_valence": data.labels["valence"][test_indices],
        "feature_names": np.asarray(data.feature_names),
    }
    if log_variance is not None:
        payload["standardized_log_variance"] = log_variance.astype(np.float32)
    np.savez_compressed(root / f"subject_{int(held):02d}_{budget}_{method}.npz", **payload)


def run_loso(
    data: ReconstructionData,
    montages: dict[str, list[str]],
    methods: list[str],
    config: ModelConfig,
    device: str,
    max_subjects: int | None = None,
    seed: int = 10,
    prediction_root: str | Path | None = None,
) -> ExperimentResults:
    all_subjects = np.unique(data.subjects)
    held_subjects = all_subjects[:max_subjects] if max_subjects is not None else all_subjects
    reconstruction_rows, group_rows, downstream_rows = [], [], []
    progress = tqdm(
        total=len(held_subjects) * len(montages) * len(methods),
        desc=f"{data.dataset} LOSO",
        unit="run",
        dynamic_ncols=True,
    )
    try:
        for held in held_subjects:
            available = all_subjects[all_subjects != held]
            offset = (seed + int(held)) % len(available)
            validation_subjects = np.roll(available, -offset)[: min(config.validation_subjects, len(available) - 1)]
            train_mask = np.isin(data.subjects, available[~np.isin(available, validation_subjects)])
            valid_mask = np.isin(data.subjects, validation_subjects)
            test_mask = data.subjects == held
            train_indices, valid_indices, test_indices = map(np.flatnonzero, (train_mask, valid_mask, test_mask))
            classifier_indices = np.concatenate([train_indices, valid_indices])
            for budget, regions in montages.items():
                raw_train = source_tensor(data, regions, train_indices)
                raw_valid = source_tensor(data, regions, valid_indices)
                raw_test = source_tensor(data, regions, test_indices)
                raw_classifier = source_tensor(data, regions, classifier_indices)
                sx, sy = FeatureScaler().fit(raw_train), FeatureScaler().fit(data.target[train_indices])
                x_train, x_valid, x_test = map(sx.transform, (raw_train, raw_valid, raw_test))
                y_train_s, y_valid_s = sy.transform(data.target[train_indices]), sy.transform(data.target[valid_indices])
                mlp_base_model = None
                if any(method in methods for method in NEURAL_METHODS):
                    mlp_base_model = _train_neural(
                        "mlp", x_train, y_train_s, x_valid, y_valid_s, config, device, seed + int(held)
                    )
                for method in methods:
                    progress.set_postfix(subject=int(held), montage=budget, method=method)
                    residual_scale, variance_offset, base_mlp_weight = float("nan"), float("nan"), float("nan")
                    if method == "ridge":
                        model = Ridge(alpha=10.0).fit(x_train.reshape(len(x_train), -1), y_train_s)
                        predict = lambda values: (model.predict(values.reshape(len(values), -1)), None)
                    elif method == "interpolation":
                        predict = lambda values, regions=regions: (_geometry_interpolation(values, regions), None)
                    elif method in NEURAL_METHODS:
                        if method in REGIONAL_METHODS:
                            if mlp_base_model is None:
                                raise RuntimeError("The global MLP base was not initialized.")
                            base_train, _ = _predict_neural(mlp_base_model, "mlp", x_train, device)
                            base_valid, _ = _predict_neural(mlp_base_model, "mlp", x_valid, device)
                            interpolation_train = _geometry_interpolation(x_train, regions)
                            interpolation_valid = _geometry_interpolation(x_valid, regions)
                            if method == "regional_no_geo":
                                base_mlp_weight = 1.0
                            else:
                                base_mlp_weight = _select_base_blend(
                                    y_valid_s,
                                    interpolation_valid,
                                    base_valid,
                                    config.correlation_weight,
                                )
                            base_train = (
                                (1.0 - base_mlp_weight) * interpolation_train + base_mlp_weight * base_train
                            )
                            base_valid = (
                                (1.0 - base_mlp_weight) * interpolation_valid + base_mlp_weight * base_valid
                            )
                            training_config = replace(config, nll_weight=0.0) if method == "regional_no_uncertainty" else config
                            model = _train_neural(
                                method,
                                x_train,
                                y_train_s,
                                x_valid,
                                y_valid_s,
                                training_config,
                                device,
                                seed + int(held),
                                base_train,
                                base_valid,
                            )
                            validation_residual, validation_logvar = _predict_neural(model, "regional", x_valid, device)
                            if method == "regional_no_gate":
                                residual_scale = 1.0
                            else:
                                residual_scale = _select_residual_scale(
                                    y_valid_s, base_valid, validation_residual, config.correlation_weight
                                )
                            if method != "regional_no_uncertainty":
                                validation_error = y_valid_s - (base_valid + residual_scale * validation_residual)
                                variance_offset = float(
                                    np.clip(
                                        np.log(np.mean(validation_error**2 * np.exp(-validation_logvar)) + 1e-8),
                                        -4.0,
                                        4.0,
                                    )
                                )
                            else:
                                variance_offset = 0.0

                            def predict(
                                values,
                                model=model,
                                mlp_base_model=mlp_base_model,
                                residual_scale=residual_scale,
                                variance_offset=variance_offset,
                                base_mlp_weight=base_mlp_weight,
                                has_uncertainty=method != "regional_no_uncertainty",
                                regions=regions,
                            ):
                                residual, log_variance = _predict_neural(model, "regional", values, device)
                                mlp_base, _ = _predict_neural(mlp_base_model, "mlp", values, device)
                                interpolation_base = _geometry_interpolation(values, regions)
                                base = (
                                    (1.0 - base_mlp_weight) * interpolation_base
                                    + base_mlp_weight * mlp_base
                                )
                                if method == "regional_no_uncertainty":
                                    return base + residual_scale * residual, log_variance
                                if not has_uncertainty:
                                    return base + residual_scale * residual, None
                                return base + residual_scale * residual, log_variance + variance_offset
                        elif method == "mc_dropout":
                            if mlp_base_model is None:
                                raise RuntimeError("The MLP model was not initialized.")
                            validation_mean, validation_logvar = _predict_mc_dropout(
                                mlp_base_model, x_valid, device, config.mc_samples
                            )
                            validation_error = y_valid_s - validation_mean
                            variance_offset = float(
                                np.clip(
                                    np.log(np.mean(validation_error**2 * np.exp(-validation_logvar)) + 1e-8),
                                    -4.0,
                                    4.0,
                                )
                            )

                            def predict(
                                values,
                                model=mlp_base_model,
                                variance_offset=variance_offset,
                            ):
                                mean, log_variance = _predict_mc_dropout(
                                    model, values, device, config.mc_samples
                                )
                                return mean, log_variance + variance_offset
                        else:
                            if mlp_base_model is None:
                                raise RuntimeError("The MLP model was not initialized.")
                            predict = lambda values, model=mlp_base_model: _predict_neural(model, "mlp", values, device)
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    predicted_test_s, test_logvar = predict(x_test)
                    predicted_classifier_s, _ = predict(sx.transform(raw_classifier))
                    predicted_test = sy.inverse_transform(predicted_test_s)
                    predicted_classifier = sy.inverse_transform(predicted_classifier_s)
                    y_test_s = sy.transform(data.target[test_indices])
                    nll = None
                    if test_logvar is not None:
                        nll = float(np.mean(0.5 * (test_logvar + (y_test_s - predicted_test_s) ** 2 * np.exp(-test_logvar))))
                    metadata = {
                        "dataset": data.dataset,
                        "held_out_subject": int(held),
                        "budget": budget,
                        "method": method,
                        "regions": ",".join(regions),
                        "residual_scale": residual_scale,
                        "variance_offset": variance_offset,
                        "base_mlp_weight": base_mlp_weight,
                    }
                    reconstruction_metrics = _metrics(data.target[test_indices], predicted_test, nll)
                    if test_logvar is not None:
                        reconstruction_metrics.update(_uncertainty_metrics(y_test_s, predicted_test_s, test_logvar))
                    reconstruction_rows.append({**metadata, **reconstruction_metrics})
                    group_rows.extend(_feature_group_rows(data, data.target[test_indices], predicted_test, metadata))
                    downstream_rows.extend(
                        _downstream_rows(
                            data,
                            classifier_indices,
                            test_indices,
                            raw_classifier,
                            raw_test,
                            predicted_classifier,
                            predicted_test,
                            metadata,
                            seed + int(held),
                        )
                    )
                    if prediction_root is not None and method in (REGIONAL_METHODS | {"mc_dropout"}):
                        _save_fold_predictions(
                            Path(prediction_root),
                            data,
                            int(held),
                            budget,
                            method,
                            test_indices,
                            predicted_test,
                            y_test_s,
                            predicted_test_s,
                            test_logvar,
                        )
                    progress.update(1)
    finally:
        progress.close()
    return ExperimentResults(
        pd.DataFrame(reconstruction_rows), pd.DataFrame(group_rows), pd.DataFrame(downstream_rows)
    )


def write_results(results: ExperimentResults, root: str | Path, dataset: str, run_name: str) -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "reconstruction": results.reconstruction,
        "feature_groups": results.feature_groups,
        "downstream": results.downstream,
    }
    paths = []
    for name, frame in tables.items():
        per_subject = root / f"{dataset}_{run_name}_{name}_per_subject.csv"
        summary = root / f"{dataset}_{run_name}_{name}_summary.csv"
        frame.to_csv(per_subject, index=False)
        group_columns = {
            "reconstruction": ["dataset", "budget", "method"],
            "feature_groups": ["dataset", "budget", "method", "feature_group"],
            "downstream": ["dataset", "budget", "method", "target", "representation"],
        }[name]
        numeric = frame.select_dtypes(include=np.number).columns.difference(["held_out_subject", "n_test_trials"])
        frame.groupby(group_columns)[list(numeric)].agg(["mean", "std", "count"]).to_csv(summary)
        paths.extend([per_subject, summary])
    return paths
