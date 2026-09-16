from __future__ import annotations

import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch

from external_wearable_dcs_opct_core import (
    ExternalApplicationConfig,
    attach_registered_outcomes,
    evaluate_locked_endpoint,
    fit_unlabeled_action,
    load_unlabeled_configuration_table,
    sha256,
    write_blinded_action_bundle,
)
from develop_distribution_covered_stratified_opct_v11 import (
    cluster_geometry,
    select_distribution_covered_audit,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
RISK_ROOT = ROOT / "outputs/identity_shortcut_risk_classifier"


def synthetic_tables(
    root: Path, config: ExternalApplicationConfig
) -> tuple[Path, Path, pd.DataFrame]:
    summary_rows = []
    encoding_rows = []
    for representation_index, representation in enumerate(config.representations):
        for seed_index, split_seed in enumerate(config.seeds):
            for probe_model, offset in (("linear", 0.0), ("rbf", 0.01)):
                encoding_rows.append(
                    {
                        "dataset": config.dataset,
                        "representation": representation,
                        "split_seed": split_seed,
                        "model": probe_model,
                        "n_features": 24 + representation_index * 12,
                        "encoding_margin": 0.15 + 0.01 * seed_index + offset,
                    }
                )
            for task_index, task in enumerate(config.tasks):
                for model_index, model in enumerate(config.models):
                    for nominal_dose in config.doses:
                        summary_rows.append(
                            {
                                "dataset": config.dataset,
                                "task": task,
                                "axis": config.axis,
                                "representation": representation,
                                "model": model,
                                "split_seed": split_seed,
                                "nominal_dose": nominal_dose,
                                "achieved_opportunity": 0.03
                                + 0.10 * nominal_dose
                                + 0.002 * task_index
                                + 0.001 * seed_index,
                                "metadata_prior_opportunity": 0.02
                                + 0.06 * nominal_dose
                                + 0.001 * representation_index,
                                "exposure_effect": (
                                    0.0
                                    if nominal_dose == 0.0
                                    else (0.03 if nominal_dose >= 0.75 else 0.01)
                                    + 0.001 * model_index
                                ),
                            }
                        )
    mechanism_path = root / "mechanism_summary.csv"
    encoding_path = root / "identity_encoding_margin.csv"
    full_summary = pd.DataFrame(summary_rows)
    full_summary.drop(columns="exposure_effect").to_csv(mechanism_path, index=False)
    pd.DataFrame(encoding_rows).to_csv(encoding_path, index=False)
    return mechanism_path, encoding_path, full_summary


def manual_endpoint_table(config: ExternalApplicationConfig) -> pd.DataFrame:
    rows = []
    identity = (0.35, 0.45, 0.55, 0.65)
    action = (0.15, 0.30, 0.70, 0.85)
    event = (0, 0, 1, 1)
    for task in config.tasks:
        for representation in config.representations:
            for model in config.models:
                for split_seed in config.seeds:
                    for index, nominal_dose in enumerate(config.doses[1:]):
                        rows.append(
                            {
                                "dataset": config.dataset,
                                "task": task,
                                "axis": config.axis,
                                "representation": representation,
                                "model": model,
                                "split_seed": split_seed,
                                "nominal_dose": nominal_dose,
                                "probability_identity": identity[index],
                                "probability_wg_opct": action[index],
                                "material_optimism_event": event[index],
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the external DCS-OPCT dry run")
    config = ExternalApplicationConfig(
        dataset="SYNTHETIC-WEARABLE",
        tasks=("valence", "arousal"),
        representations=("all", "relative_power", "normalized_asymmetry"),
        models=("linear_logistic", "gpu_mlp"),
        seeds=(20260813, 20260829, 20260911, 20260923, 20261007),
    )
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    risk_manifest = json.loads(
        (RISK_ROOT / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    risk_model = joblib.load(RISK_ROOT / "frozen_risk_model.joblib")
    public = pd.read_csv(RISK_ROOT / "risk_learning_table.csv")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mechanism_path, encoding_path, full_summary = synthetic_tables(root, config)
        target = load_unlabeled_configuration_table(
            mechanism_path, encoding_path, config
        )
        assert len(target) == 240
        assert not {
            "exposure_effect",
            "dose_induced_amplification",
            "material_optimism_event",
        } & set(target)
        leaked_target = target.copy()
        leaked_target["material_optimism_event"] = 0
        try:
            fit_unlabeled_action(
                leaked_target,
                public,
                risk_model,
                risk_manifest,
                freeze,
                config,
                device="cuda",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("The unlabeled stage accepted a material-event column")
        prepared = fit_unlabeled_action(
            target,
            public,
            risk_model,
            risk_manifest,
            freeze,
            config,
            device="cuda",
        )
        assert len(prepared["mechanism"]) == 240
        assert len(prepared["geometry"]) == 60
        assert prepared["assignment"].partition.value_counts().to_dict() == {
            "audit": 30,
            "heldout": 30,
        }
        lock_path, lock_hash = write_blinded_action_bundle(
            root / "application",
            prepared,
            {
                "freeze_sha256": sha256(FREEZE),
                "mechanism_summary_sha256": sha256(mechanism_path),
            },
        )
        summary_path = root / "summary.csv"
        full_summary.to_csv(summary_path, index=False)
        evaluation = attach_registered_outcomes(
            prepared["mechanism"], summary_path, config, lock_path, lock_hash
        )
        assert len(evaluation) == 240
        assert set(evaluation.material_optimism_event) == {0, 1}
        result = evaluate_locked_endpoint(
            evaluation,
            prepared["assignment"],
            prepared["components"],
            bool(prepared["candidate_applicable"]),
            freeze,
        )
        assert result["endpoint_estimable"] is True
        if result["selected_method"] == "identity":
            assert result["effectiveness_pass"] is False
            assert result["released_action_nonharm_pass"] is None
            assert result["safety_success"] is False

        tampered_lock = root / "tampered_action_lock.json"
        tampered_lock.write_bytes(lock_path.read_bytes() + b"\n")
        try:
            attach_registered_outcomes(
                prepared["mechanism"],
                summary_path,
                config,
                tampered_lock,
                lock_hash,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("Outcome attachment accepted a modified action lock")

    manual = manual_endpoint_table(config)
    assignment = select_distribution_covered_audit(cluster_geometry(manual))
    components = pd.DataFrame(
        [
            {
                "method": "coral",
                "applicable": True,
                "probability_rank": 1.0,
                "order_inversions": 0,
            },
            {
                "method": "quantile_mapping",
                "applicable": True,
                "probability_rank": 1.0,
                "order_inversions": 0,
            },
        ]
    )
    released = evaluate_locked_endpoint(manual, assignment, components, True, freeze)
    assert released["endpoint_estimable"] is True
    assert released["selected_method"] == "wg_opct"
    assert released["effectiveness_pass"] is True
    assert released["released_action_nonharm_pass"] is True
    assert released["claim_supported"] is True

    no_events = manual.copy()
    no_events["material_optimism_event"] = 0
    non_estimable = evaluate_locked_endpoint(
        no_events, assignment, components, True, freeze
    )
    assert non_estimable["endpoint_estimable"] is False
    assert non_estimable["selected_method"] == "identity"
    assert non_estimable["heldout_metrics"] is None
    assert non_estimable["safety_success"] is False
    print(
        "External wearable DCS-OPCT CUDA dry run passed: "
        "blind lock, estimable release, identity boundary, and zero-event boundary"
    )


if __name__ == "__main__":
    main()
