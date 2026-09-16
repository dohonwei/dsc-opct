from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from apply_frozen_scmt_to_eppvr import paired_bootstrap, performance  # noqa: E402
from identity_shortcut.domain_transport import (  # noqa: E402
    SCMTConfig,
    coral,
    fit_scmt,
    mean_shift,
    quantile_map,
    support_clip,
)
from identity_shortcut.transport_policy import (  # noqa: E402
    PolicyConfig,
    select_actions,
    unlabeled_signature,
)


SCMT_SEED = 20260905


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrospectively apply the frozen unlabeled transport policy to EPPVR."
    )
    parser.add_argument(
        "--risk-root", type=Path, default=Path("outputs/identity_shortcut_risk_classifier")
    )
    parser.add_argument(
        "--scmt-root", type=Path, default=Path("outputs/scmt_public_development")
    )
    parser.add_argument(
        "--policy-root", type=Path, default=Path("outputs/transport_policy_public_development")
    )
    parser.add_argument(
        "--external-root", type=Path, default=Path("outputs/eppvr_frozen_risk_validation")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/eppvr_transport_policy_retrospective")
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probability(logistic, values: np.ndarray) -> np.ndarray:
    return logistic.predict_proba(np.asarray(values))[:, 1]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required. Use --allow-cpu only for debugging.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = {
        "risk_manifest": args.risk_root / "freeze_manifest.json",
        "risk_model": args.risk_root / "frozen_risk_model.joblib",
        "risk_table": args.risk_root / "risk_learning_table.csv",
        "scmt_manifest": args.scmt_root / "scmt_freeze_manifest.json",
        "policy_manifest": args.policy_root / "transport_policy_freeze_manifest.json",
        "policy_artifact": args.policy_root / "frozen_transport_policy.joblib",
        "target_predictions": args.external_root / "eppvr_risk_predictions.csv",
        "locked_gate": args.external_root / "external_validation_gate.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    risk_manifest = json.loads(paths["risk_manifest"].read_text(encoding="utf-8"))
    scmt_manifest = json.loads(paths["scmt_manifest"].read_text(encoding="utf-8"))
    policy_manifest = json.loads(paths["policy_manifest"].read_text(encoding="utf-8"))
    if not policy_manifest.get("admissible"):
        raise ValueError("The public transport policy did not pass its frozen gate")
    if sha256(paths["risk_model"]) != risk_manifest["model_sha256"]:
        raise ValueError("Frozen risk-model hash mismatch")
    if sha256(paths["scmt_manifest"]) != policy_manifest["source_scmt_manifest_sha256"]:
        raise ValueError("Transport policy does not reference the current SCMT freeze")
    if sha256(paths["policy_artifact"]) != policy_manifest["policy_artifact_sha256"]:
        raise ValueError("Frozen transport-policy artifact hash mismatch")
    if sha256(ROOT / "src" / "identity_shortcut" / "transport_policy.py") != policy_manifest[
        "policy_code_sha256"
    ]:
        raise ValueError("Transport-policy implementation changed after freeze")
    locked_gate_hash_before = sha256(paths["locked_gate"])
    features = policy_manifest["features"]
    public = pd.read_csv(
        paths["risk_table"], usecols=features + ["material_optimism_event"]
    )
    # Target outcomes are intentionally unavailable until the policy action is fixed.
    target_features = pd.read_csv(paths["target_predictions"], usecols=features)
    risk_model = joblib.load(paths["risk_model"])
    scaler = risk_model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = risk_model.named_steps["model"]
    source = scaler.transform(public[features]).astype(np.float32)
    source_labels = public.material_optimism_event.to_numpy(int)
    target = scaler.transform(target_features[features]).astype(np.float32)
    scmt_config = SCMTConfig(**scmt_manifest["selected_config"])
    progress = tqdm(total=1, desc="EPPVR frozen-policy SCMT candidate", unit="fit", dynamic_ncols=True)
    transformed_scmt, scmt_diagnostics = fit_scmt(
        source,
        target,
        lambda x: probability(logistic, x),
        scmt_config,
        SCMT_SEED,
        device,
        lambda epoch, loss: progress.set_postfix(
            loss=f"{loss['total']:.3f}", refresh=False
        ) if epoch % 40 == 0 else None,
    )
    progress.update(1)
    progress.close()
    transforms = {
        "identity": target.copy(),
        "mean_shift": mean_shift(source, target),
        "coral": coral(source, target),
        "quantile_mapping": quantile_map(source, target),
        "support_clipping": support_clip(source, target),
        "scmt": transformed_scmt,
    }
    threshold = float(risk_manifest["decision_threshold"])
    signatures = pd.DataFrame([
        {
            "scenario_id": "EPPVR",
            "method": method,
            **unlabeled_signature(
                source,
                source_labels,
                target,
                transformed,
                lambda x: probability(logistic, x),
                threshold,
                args.bootstrap_seed,
            ),
        }
        for method, transformed in transforms.items()
    ])
    artifact = joblib.load(paths["policy_artifact"])
    policy_config = PolicyConfig(**artifact["policy_config"])
    decision = select_actions(
        signatures,
        artifact["gain_committee"],
        artifact["auc_committee"],
        policy_config,
    )
    selected_method = str(decision.iloc[0].selected_method)
    selection_reason = str(decision.iloc[0].selection_reason)

    # Labels and configuration metadata enter only after the decision is immutable in memory.
    evaluation = pd.read_csv(paths["target_predictions"])
    labels = evaluation.material_optimism_event.to_numpy(int)
    null_probability = float(risk_manifest["public_training_event_prevalence"])
    probabilities = {
        method: probability(logistic, transformed)
        for method, transformed in transforms.items()
    }
    metrics = pd.DataFrame([
        {
            "method": method,
            "selected_by_policy": method == selected_method,
            **performance(labels, method_probability, threshold, null_probability),
        }
        for method, method_probability in probabilities.items()
    ])
    for method, method_probability in probabilities.items():
        evaluation[f"probability_{method}"] = method_probability
    intervals = paired_bootstrap(
        evaluation,
        probabilities,
        threshold,
        null_probability,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    signatures.to_csv(args.output_root / "unlabeled_candidate_signatures.csv", index=False)
    decision.to_csv(args.output_root / "frozen_policy_decision.csv", index=False)
    metrics.to_csv(args.output_root / "method_performance.csv", index=False)
    intervals.to_csv(args.output_root / "paired_bootstrap_differences.csv", index=False)
    evaluation.to_csv(args.output_root / "eppvr_policy_predictions.csv", index=False)
    pd.DataFrame([{
        key: value for key, value in scmt_diagnostics.items() if not isinstance(value, dict)
    } | {
        f"check_{key}": value for key, value in scmt_diagnostics["checks"].items()
    }]).to_csv(args.output_root / "scmt_candidate_diagnostics.csv", index=False)
    selected_metrics = metrics.loc[metrics.method == selected_method].iloc[0]
    manifest = {
        "status": "post_external_retrospective_policy_evaluation",
        "selected_method": selected_method,
        "selection_reason": selection_reason,
        "label_blinding_contract": (
            "The policy action was selected from public source labels, public source features, "
            "and unlabeled EPPVR features before EPPVR outcomes were loaded."
        ),
        "claim_boundary": (
            "EPPVR outcomes informed earlier SCMT development. This application is retrospective "
            "and cannot serve as independent validation of the transport policy."
        ),
        "selected_brier_score": float(selected_metrics.brier_score),
        "selected_roc_auc": float(selected_metrics.roc_auc),
        "selected_balanced_accuracy": float(selected_metrics.balanced_accuracy),
        "policy_manifest_sha256": sha256(paths["policy_manifest"]),
        "policy_artifact_sha256": sha256(paths["policy_artifact"]),
        "locked_external_gate_sha256_before": locked_gate_hash_before,
        "locked_external_gate_sha256_after": sha256(paths["locked_gate"]),
        "locked_external_gate_unchanged": locked_gate_hash_before == sha256(paths["locked_gate"]),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_unit": "task x representation x model x split_seed configuration",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scmt_seed": SCMT_SEED,
    }
    (args.output_root / "retrospective_policy_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(decision[[
        "selected_method",
        "selection_reason",
        "estimated_label_shift",
        "predicted_brier_gain_lcb",
        "predicted_auc_delta_lcb",
    ]].to_string(index=False))
    print(metrics.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
