from __future__ import annotations

import argparse
from pathlib import Path

from ekmed_v11_adapter_contract import validate_ekmed_task_bundle
from ekmed_v11_contract import (
    DEFAULT_ARCHIVE,
    DOSES,
    GPU_EPOCHS,
    MATERIAL_THRESHOLD,
    MODELS,
    REPRESENTATIONS,
    ROOT,
    SEEDS,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
)
from ekmed_v11_schema_adapter import load_ekmed_task_bundles
from external_wearable_dcs_opct_core import ExternalApplicationConfig
from external_wearable_dose_core import DoseRunConfig
from external_wearable_implementation_lock import sha256, verify_implementation_lock
from external_wearable_one_shot_core import (
    finalize_registered_outcomes,
    lock_registered_action,
    prepare_and_lock_preoutcome,
    run_registered_outcomes,
)


SCHEMA_MANIFEST = ROOT / "outputs/ekmed_v11_archive_schema/header_schema_receipt.json"
IMPLEMENTATION_LOCK = ROOT / "docs/ekmed_v11_external_confirmation_implementation_lock.json"
OUTPUT_ROOT = ROOT / "outputs/ekmed_v11_external_confirmation"
RISK_ROOT = ROOT / "outputs/identity_shortcut_risk_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked EKM-ED DCS-OPCT v11 one-shot GPU confirmation"
    )
    parser.add_argument(
        "stage", choices=("prepare", "action", "outcomes", "finalize", "all")
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--schema-manifest", type=Path, default=SCHEMA_MANIFEST)
    parser.add_argument("--implementation-lock", type=Path, default=IMPLEMENTATION_LOCK)
    parser.add_argument("--implementation-lock-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preoutcome-lock-sha256")
    parser.add_argument("--action-lock-sha256")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def application_config() -> ExternalApplicationConfig:
    return ExternalApplicationConfig(
        dataset="EmoKey Moments Muse EEG Dataset (EKM-ED)",
        tasks=("valence", "arousal"),
        representations=REPRESENTATIONS,
        models=MODELS,
        seeds=SEEDS,
        doses=DOSES,
        material_threshold=MATERIAL_THRESHOLD,
    )


def dose_configs(config: ExternalApplicationConfig) -> dict[str, DoseRunConfig]:
    return {
        task: DoseRunConfig(
            dataset=config.dataset,
            task=task,
            subject_folds=SUBJECT_FOLDS,
            stimulus_folds=STIMULUS_FOLDS,
            seeds=SEEDS,
            doses=DOSES,
            representations=REPRESENTATIONS,
            models=MODELS,
            gpu_epochs=GPU_EPOCHS,
            candidate_draws=256,
        )
        for task in config.tasks
    }


def require_hash(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required for this stage")
    return value


def main() -> None:
    args = parse_args()
    implementation = verify_implementation_lock(
        args.implementation_lock,
        root=ROOT,
        expected_sha256=args.implementation_lock_sha256,
    )
    locked_schema = (ROOT / implementation["schema_manifest_path"]).resolve()
    if args.schema_manifest.resolve() != locked_schema:
        raise RuntimeError(
            "The requested schema receipt is not bound to the implementation lock"
        )
    if sha256(args.schema_manifest) != implementation["schema_manifest_sha256"]:
        raise RuntimeError("The locked EKM-ED schema receipt changed")

    config = application_config()
    preoutcome_sha = args.preoutcome_lock_sha256
    action_sha = args.action_lock_sha256

    if args.stage in {"prepare", "all"}:
        bundles, adapter_provenance = load_ekmed_task_bundles(
            args.archive, args.schema_manifest
        )
        preoutcome_path, preoutcome_sha = prepare_and_lock_preoutcome(
            args.output_root,
            task_bundles=bundles,
            task_validators={
                "valence": validate_ekmed_task_bundle,
                "arousal": validate_ekmed_task_bundle,
            },
            dose_configs=dose_configs(config),
            application_config=config,
            implementation_lock_path=args.implementation_lock,
            implementation_lock_sha256=args.implementation_lock_sha256,
            project_root=ROOT,
            provenance={
                "archive": args.archive.resolve().as_posix(),
                "archive_sha256": sha256(args.archive),
                "schema_receipt": args.schema_manifest.resolve().as_posix(),
                "adapter": adapter_provenance,
            },
        )
        print(f"Pre-outcome design locked: {preoutcome_path} ({preoutcome_sha})")
        if args.stage == "prepare":
            return

    preoutcome_sha = require_hash(preoutcome_sha, "--preoutcome-lock-sha256")
    if args.stage in {"action", "all"}:
        action_path, action_sha = lock_registered_action(
            args.output_root,
            preoutcome_lock_sha256=preoutcome_sha,
            implementation_lock_path=args.implementation_lock,
            implementation_lock_sha256=args.implementation_lock_sha256,
            project_root=ROOT,
            risk_root=RISK_ROOT,
            device=args.device,
        )
        print(f"Unlabeled action locked: {action_path} ({action_sha})")
        if args.stage == "action":
            return

    action_sha = require_hash(action_sha, "--action-lock-sha256")
    summary_path = args.output_root / "model_outcomes/summary.csv"
    if args.stage in {"outcomes", "all"}:
        summary_path = run_registered_outcomes(
            args.output_root,
            preoutcome_lock_sha256=preoutcome_sha,
            action_lock_sha256=action_sha,
            implementation_lock_path=args.implementation_lock,
            implementation_lock_sha256=args.implementation_lock_sha256,
            project_root=ROOT,
            n_jobs=args.n_jobs,
            device=args.device,
        )
        print(f"Registered model outcomes complete: {summary_path}")
        if args.stage == "outcomes":
            return

    gate = finalize_registered_outcomes(
        args.output_root,
        preoutcome_lock_sha256=preoutcome_sha,
        action_lock_sha256=action_sha,
        full_summary_path=summary_path,
        implementation_lock_path=args.implementation_lock,
        implementation_lock_sha256=args.implementation_lock_sha256,
        project_root=ROOT,
    )
    print(
        "EKM-ED one-shot final gate: "
        f"selected={gate.get('selected_method')}, "
        f"claim_supported={gate.get('claim_supported')}"
    )


if __name__ == "__main__":
    main()
