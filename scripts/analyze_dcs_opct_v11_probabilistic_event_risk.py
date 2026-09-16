from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_identity_shortcut_risk import classifier_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uncertainty-table", type=Path,
        default=Path(
            "outputs/dcs_opct_v11_crossed_cluster_endpoint_uncertainty_20260914_v2/"
            "crossed_cluster_event_uncertainty.csv"
        ),
    )
    parser.add_argument(
        "--frozen-model", type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier/frozen_risk_model.joblib"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/dcs_opct_v11_probabilistic_event_risk_20260914"),
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def fit_soft_logistic(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
    epochs: int, seed: int, progress: tqdm,
) -> tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    x = torch.as_tensor(x_train, dtype=torch.float64, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float64, device=device)
    xt = torch.as_tensor(x_test, dtype=torch.float64, device=device)
    weight = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float64, device=device))
    intercept = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device=device))
    optimizer = torch.optim.Adam([weight, intercept], lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weight + intercept
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + 0.05 * torch.sum(weight**2) / len(x_train)
        loss.backward()
        optimizer.step()
        progress.update(1)
    probability = torch.sigmoid(xt @ weight + intercept).detach().cpu().numpy()
    return probability, weight.detach().cpu().numpy(), float(loss.detach().cpu())


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    table = pd.read_csv(args.uncertainty_table)
    frozen = joblib.load(args.frozen_model)
    features = classifier_features()
    rows = []
    coefficient_rows = []
    prediction_rows = []
    progress = tqdm(
        total=2 * args.epochs, desc="GPU probabilistic-event LODO fitting",
        unit="epoch", dynamic_ncols=True,
    )
    for fold_index, held_out in enumerate(sorted(table.dataset.unique())):
        train = table.loc[table.dataset.ne(held_out)].copy()
        test = table.loc[table.dataset.eq(held_out)].copy()
        scaler = StandardScaler().fit(train[features])
        probability, coefficients, loss = fit_soft_logistic(
            scaler.transform(train[features]),
            train.probability_material_event.to_numpy(float),
            scaler.transform(test[features]),
            args.epochs,
            args.seed + fold_index * 7919,
            progress,
        )
        frozen_probability = frozen.predict_proba(test[features])[:, 1]
        soft = test.probability_material_event.to_numpy(float)
        hard = test.material_optimism_event.to_numpy(int)
        rows.append(
            {
                "held_out": held_out,
                "n_test": len(test),
                "soft_prevalence": float(soft.mean()),
                "hard_prevalence": float(hard.mean()),
                "probabilistic_model_soft_brier": float(np.mean((probability - soft) ** 2)),
                "frozen_hard_model_soft_brier": float(np.mean((frozen_probability - soft) ** 2)),
                "probabilistic_model_hard_auroc": float(roc_auc_score(hard, probability)),
                "frozen_hard_model_hard_auroc": float(roc_auc_score(hard, frozen_probability)),
                "probability_spearman": float(spearmanr(probability, frozen_probability).statistic),
                "final_training_loss": loss,
            }
        )
        for feature, value in zip(features, coefficients, strict=True):
            coefficient_rows.append(
                {"held_out": held_out, "feature": feature, "coefficient": float(value)}
            )
        for row_id, soft_label, estimate, frozen_estimate in zip(
            test.risk_row_id, soft, probability, frozen_probability, strict=True
        ):
            prediction_rows.append(
                {
                    "held_out": held_out,
                    "risk_row_id": int(row_id),
                    "probabilistic_event_label": float(soft_label),
                    "probability_soft_model": float(estimate),
                    "probability_frozen_hard_model": float(frozen_estimate),
                }
            )
    progress.close()
    metrics = pd.DataFrame(rows)
    coefficients = pd.DataFrame(coefficient_rows)
    predictions = pd.DataFrame(prediction_rows)
    args.output_root.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(args.output_root / "probabilistic_event_lodo_metrics.csv", index=False)
    coefficients.to_csv(args.output_root / "probabilistic_event_coefficients.csv", index=False)
    predictions.to_csv(args.output_root / "probabilistic_event_predictions.csv", index=False)
    report = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "soft_label": "crossed-bootstrap probability that dose-induced BAcc amplification is at least 0.02",
        "claim_boundary": "Post-hoc measurement-error sensitivity; it does not replace the frozen binary endpoint or create an externally validated operating threshold.",
    }
    (args.output_root / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(coefficients.to_string(index=False))


if __name__ == "__main__":
    main()
