from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    CLUSTER_COLUMNS,
    partition_frame,
    stratified_certificate,
)
from develop_order_preserving_transport_v6 import RULE  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import (  # noqa: E402
    load_development,
    prepare as prepare_v9,
    witness_statistics,
)
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402


OUT = ROOT / "outputs/dcs_opct_v11_leave_one_component"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
DEV = ROOT / "outputs/distribution_covered_stratified_opct_v11_development"
V9 = ROOT / "outputs/witness_gated_covariance_opct_v9_development"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
SEED = 20260908


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_frozen_inventory() -> dict:
    if sha256(FREEZE) != EXPECTED_FREEZE_HASH:
        raise RuntimeError("Frozen v11 manifest hash changed")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    mismatches = {}
    for relative, expected in freeze["locked_artifacts"].items():
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"Frozen artifact mismatch: {mismatches}")
    return freeze


def risk_config() -> RiskControlledConfig:
    return RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=5000,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=500,
    )


def verify_reconstruction(components: pd.DataFrame, locks: pd.DataFrame) -> None:
    frozen_components = pd.read_csv(DEV / "component_projection_diagnostics.csv")
    frozen_locks = pd.read_csv(DEV / "unlabeled_action_locks.csv")
    component_keys = ["dataset", "method"]
    lock_keys = ["dataset", "method"]
    components = components.sort_values(component_keys).reset_index(drop=True)
    frozen_components = frozen_components.sort_values(component_keys).reset_index(drop=True)
    locks = locks.sort_values(lock_keys).reset_index(drop=True)
    frozen_locks = frozen_locks.sort_values(lock_keys).reset_index(drop=True)
    if components[component_keys].to_dict("records") != frozen_components[component_keys].to_dict("records"):
        raise RuntimeError("Reconstructed component keys do not match frozen diagnostics")
    if locks[lock_keys].to_dict("records") != frozen_locks[lock_keys].to_dict("records"):
        raise RuntimeError("Reconstructed lock keys do not match frozen action locks")
    for current, frozen, name in (
        (components, frozen_components, "components"),
        (locks, frozen_locks, "locks"),
    ):
        common_numeric = [
            column
            for column in current.columns.intersection(frozen.columns)
            if pd.api.types.is_numeric_dtype(current[column])
            and pd.api.types.is_numeric_dtype(frozen[column])
        ]
        for column in common_numeric:
            if not np.allclose(
                current[column].to_numpy(float),
                frozen[column].to_numpy(float),
                rtol=0,
                atol=1e-8,
                equal_nan=True,
            ):
                raise RuntimeError(f"Reconstructed {name}.{column} differs from frozen output")


def candidate_diagnostics(frame: pd.DataFrame) -> dict[str, float | int]:
    identity = frame.probability_identity.to_numpy(float)
    candidate = frame.probability_component_coral.to_numpy(float)
    rank = float(pd.Series(identity).rank(method="average").corr(pd.Series(candidate).rank(method="average")))
    inversions = int(
        np.sum(
            np.sign(identity[:, None] - identity[None, :])
            * np.sign(candidate[:, None] - candidate[None, :])
            < 0
        )
        // 2
    )
    return {"rank": rank, "inversions": inversions}


def evaluate_variant(
    dataset: str,
    frame: pd.DataFrame,
    assignment: pd.DataFrame,
    config: RiskControlledConfig,
    eligible: bool,
    variant: str,
) -> dict[str, object]:
    audit, heldout = partition_frame(frame, assignment)
    diagnostics = candidate_diagnostics(frame)
    audit = audit.copy()
    heldout = heldout.copy()
    audit["probability_diagnostic"] = audit.probability_component_coral
    heldout["probability_diagnostic"] = heldout.probability_component_coral
    audit["probability_wg_opct"] = audit.probability_diagnostic
    certificate = stratified_certificate(
        audit,
        config,
        SEED + RULE.configuration_budget * 1009,
        eligible,
        diagnostics["rank"],
        diagnostics["inversions"],
    )
    selected = "diagnostic" if certificate["certified"] else "identity"
    outcome = evaluate_action(heldout, selected)
    return {
        "dataset": dataset,
        "variant": variant,
        "candidate_eligible": eligible,
        "audit_brier_gain": certificate["brier_gain"],
        "audit_brier_gain_lcb": certificate["brier_gain_lcb"],
        "audit_certified": certificate["certified"],
        "certificate_failure_reason": certificate["failure_reason"],
        "selected_action": "CORAL + OPCT" if outcome["adapted"] else "identity",
        "heldout_brier_gain": outcome["brier_gain"],
        "material_negative_transfer": outcome["material_negative_transfer"],
        "probability_rank": diagnostics["rank"],
        "order_inversions": diagnostics["inversions"],
    }


def format_cell(row: pd.Series) -> str:
    if not bool(row.candidate_eligible):
        return "I"
    if not bool(row.audit_certified):
        return "C-reject"
    return f"R ({row.heldout_brier_gain:+.4f})"


def write_latex_tables(same: pd.DataFrame, random_summary: pd.DataFrame) -> None:
    order = ["full_v11", "minus_quantile_witness", "minus_unlabeled_applicability"]
    labels = {
        "full_v11": "Full v11",
        "minus_quantile_witness": "Without quantile witness",
        "minus_unlabeled_applicability": "Without unlabeled screen",
    }
    datasets = ["EPPVR", "SEED-IV", "DREAMER", "CASE", "CEAP"]
    lines = ["\\begin{tabular}{lccccc}", "\\toprule", "Variant & " + " & ".join(datasets) + " \\\\", "\\midrule"]
    for variant in order:
        indexed = same.loc[same.variant.eq(variant)].set_index("dataset")
        cells = [format_cell(indexed.loc[dataset]) for dataset in datasets]
        lines.append(labels[variant] + " & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (OUT / "table_leave_one_component.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & Covered gain & Random release & Random mean gain & Random material harm \\\\",
        "\\midrule",
    ]
    for row in random_summary.itertuples(index=False):
        lines.append(
            f"{row.dataset} & {row.covered_heldout_brier_gain:+.4f} & "
            f"{row.random_release_rate:.2f} & {row.random_mean_heldout_brier_gain:+.4f} & "
            f"{row.random_material_negative_frequency:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (OUT / "table_distribution_coverage_diagnostic.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reconstruct the frozen OPCT probabilities")
    freeze = assert_frozen_inventory()
    config = risk_config()
    prepared, components, locks = prepare_v9(load_development(), config, SEED)
    verify_reconstruction(components, locks)
    component_index = components.set_index(["dataset", "method"])
    lock_index = locks.set_index("dataset")
    assignments = pd.read_csv(DEV / "distribution_covered_assignments.csv")

    same_rows = []
    progress = tqdm(
        total=len(prepared) * 3,
        desc="Frozen leave-one-component diagnostic",
        unit="cell",
        dynamic_ncols=True,
    )
    for dataset, frame in prepared.items():
        identity = frame.probability_identity.to_numpy(float)
        coral = frame.probability_component_coral.to_numpy(float)
        quantile = frame.probability_component_quantile_mapping.to_numpy(float)
        raw_witness = witness_statistics(identity, coral, quantile)
        coral_applicable = bool(component_index.loc[(dataset, "coral"), "applicable"])
        eligibility = {
            "full_v11": bool(lock_index.loc[dataset, "applicable"]),
            "minus_quantile_witness": coral_applicable,
            "minus_unlabeled_applicability": True,
        }
        assignment = assignments.loc[assignments.dataset.eq(dataset)].drop(columns="dataset")
        for variant, eligible in eligibility.items():
            row = evaluate_variant(dataset, frame, assignment, config, eligible, variant)
            row.update(
                {
                    "coral_component_applicable": coral_applicable,
                    "raw_quantile_witness_pass": bool(raw_witness["witness_pass"]),
                }
            )
            same_rows.append(row)
            progress.update(1)
    progress.close()
    same = pd.DataFrame(same_rows)

    random_assignments = pd.read_csv(V9 / "audit_cluster_assignments.csv")
    random_rows = []
    total = random_assignments[["dataset", "audit_repetition"]].drop_duplicates().shape[0]
    progress = tqdm(total=total, desc="Random-half allocation diagnostic", unit="audit", dynamic_ncols=True)
    for dataset, frame in prepared.items():
        eligible = bool(lock_index.loc[dataset, "applicable"])
        dataset_assignments = random_assignments.loc[random_assignments.dataset.eq(dataset)]
        for repetition, assignment in dataset_assignments.groupby("audit_repetition", sort=True):
            partition = assignment[[*CLUSTER_COLUMNS, "partition"]]
            row = evaluate_variant(
                dataset,
                frame,
                partition,
                config,
                eligible,
                "random_half_without_distribution_coverage",
            )
            row["audit_repetition"] = int(repetition)
            random_rows.append(row)
            progress.update(1)
    progress.close()
    random_results = pd.DataFrame(random_rows)

    covered = same.loc[same.variant.eq("full_v11")].set_index("dataset")
    summaries = []
    for dataset, group in random_results.groupby("dataset", sort=False):
        released = group.loc[group.audit_certified]
        summaries.append(
            {
                "dataset": dataset,
                "covered_release": bool(covered.loc[dataset, "audit_certified"]),
                "covered_heldout_brier_gain": float(covered.loc[dataset, "heldout_brier_gain"]),
                "random_repetitions": len(group),
                "random_release_rate": float(group.audit_certified.mean()),
                "random_mean_heldout_brier_gain": float(group.heldout_brier_gain.mean()),
                "random_minimum_gain_when_released": (
                    float(released.heldout_brier_gain.min()) if len(released) else np.nan
                ),
                "random_material_negative_frequency": float(group.material_negative_transfer.mean()),
            }
        )
    random_summary = pd.DataFrame(summaries)

    OUT.mkdir(parents=True, exist_ok=False)
    same.to_csv(OUT / "same_assignment_leave_one_component.csv", index=False)
    random_results.to_csv(OUT / "random_half_allocation_repetitions.csv", index=False)
    random_summary.to_csv(OUT / "distribution_coverage_summary.csv", index=False)
    write_latex_tables(same, random_summary)
    outputs = sorted(path for path in OUT.iterdir() if path.is_file())
    manifest = {
        "status": "retrospective_frozen_leave_one_component_diagnostic_complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_status": "diagnostic_only_no_gate_or_threshold_tuning",
        "claim_boundary": (
            "These counterfactual removals reuse frozen probabilities, thresholds, seeds, assignments, and held-out rows. "
            "They diagnose observed component behavior and do not establish causal necessity or prospective safety."
        ),
        "variants": {
            "full_v11": "Frozen component invariants, quantile witness, distribution-covered audit, and certificate.",
            "minus_quantile_witness": "CORAL component invariant screen and certificate retained; witness removed.",
            "minus_unlabeled_applicability": "Only the labeled certificate retained before release.",
            "random_half_without_distribution_coverage": "Full unlabeled action retained; stored v9 random half-assignments replace the v11 covered assignment.",
        },
        "frozen_rule": asdict(RULE),
        "seed": SEED,
        "bootstrap_repetitions": config.bootstrap_repetitions,
        "gpu": torch.cuda.get_device_name(0),
        "freeze_sha256": sha256(FREEZE),
        "input_sha256": {
            "v11_assignments": sha256(DEV / "distribution_covered_assignments.csv"),
            "v11_components": sha256(DEV / "component_projection_diagnostics.csv"),
            "v11_action_locks": sha256(DEV / "unlabeled_action_locks.csv"),
            "v11_certificates": sha256(DEV / "primary_certificates.csv"),
            "v11_heldout": sha256(DEV / "primary_heldout_results.csv"),
            "v9_random_assignments": sha256(V9 / "audit_cluster_assignments.csv"),
        },
        "frozen_inventory_count": len(freeze["locked_artifacts"]),
        "script_sha256": sha256(Path(__file__)),
        "output_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in outputs},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(same.to_string(index=False))
    print(random_summary.to_string(index=False))


if __name__ == "__main__":
    main()
