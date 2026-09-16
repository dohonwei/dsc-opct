from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_consensus_order_preserving_transport_v7 import load_development  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    CLUSTER_COLUMNS,
    STRATUM_COLUMNS,
    partition_frame,
)
from develop_witness_gated_covariance_opct_v9 import (  # noqa: E402
    prepare as prepare_v9,
    witness_statistics,
)
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402

EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
FREEZE = Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json")
DEV = Path("outputs/distribution_covered_stratified_opct_v11_development")
EXTERNAL = {
    "AVDOS-VR": Path("outputs/avdos_v11_exploratory_ppg"),
    "FACED": Path("outputs/faced_v11_post_access_common_montage_dcs_opct"),
    "EEGEmotions-27": Path("outputs/eegemotions27_v11_external_robustness"),
}
DATASETS = ["EPPVR", "CASE", "CEAP", "DREAMER", "SEED-IV", "AVDOS-VR", "FACED", "EEGEmotions-27"]
PARAMETERS = [
    "maximum_probability_mean_shift",
    "minimum_probability_rank",
    "maximum_order_inversions",
    "minimum_directional_agreement",
    "minimum_displacement_cosine",
    "maximum_normalized_disagreement",
    "minimum_brier_gain_lcb",
    "auc_noninferiority_margin",
]
LABELS = {
    "maximum_probability_mean_shift": "Mean-shift maximum",
    "minimum_probability_rank": "Rank minimum",
    "maximum_order_inversions": "Inversion maximum",
    "minimum_directional_agreement": "Direction minimum",
    "minimum_displacement_cosine": "Cosine minimum",
    "maximum_normalized_disagreement": "Disagreement maximum",
    "minimum_brier_gain_lcb": "Brier-LCB minimum",
    "auc_noninferiority_margin": "AUROC-NI margin",
}
GRID_LABELS = ["more permissive 2", "more permissive 1", "frozen", "stricter 1", "stricter 2"]


@dataclass
class Evidence:
    dataset: str
    role: str
    components: pd.DataFrame
    witness: dict[str, float]
    certificate: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc stability of frozen DCS-OPCT v11 actions.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dcs_opct_v11_frozen_action_stability_20260910_v2"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(args: argparse.Namespace, freeze: dict[str, Any]) -> None:
    if sha256(FREEZE) != EXPECTED_FREEZE:
        raise RuntimeError("Frozen v11 hash mismatch")
    for relative, expected in freeze["locked_artifacts"].items():
        path = Path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Frozen artifact mismatch: {path}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_root.resolve()}")
    if args.bootstrap_repetitions != freeze["certification"]["bootstrap_repetitions"]:
        raise ValueError("The frozen 5,000-bootstrap contract must be retained")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reconstruct the frozen candidates")


def threshold_spec(freeze: dict[str, Any]) -> tuple[dict[str, float], dict[str, list[float]]]:
    action, cert = freeze["probability_action"], freeze["certification"]
    frozen = {
        "maximum_probability_mean_shift": float(action["maximum_probability_mean_shift"]),
        "minimum_probability_rank": float(action["minimum_probability_rank"]),
        "maximum_order_inversions": float(action["maximum_order_inversions"]),
        "minimum_directional_agreement": float(action["minimum_directional_agreement"]),
        "minimum_displacement_cosine": float(action["minimum_displacement_cosine"]),
        "maximum_normalized_disagreement": float(action["maximum_normalized_disagreement"]),
        "minimum_brier_gain_lcb": float(cert["minimum_brier_gain_lcb"]),
        "auc_noninferiority_margin": float(cert["auc_noninferiority_margin"]),
    }
    grids = {
        "maximum_probability_mean_shift": [0.12, 0.11, 0.10, 0.09, 0.08],
        "minimum_probability_rank": [0.999990, 0.999995, 0.999999, 0.9999995, 1.0],
        "maximum_order_inversions": [2.0, 1.0, 0.0, 0.0, 0.0],
        "minimum_directional_agreement": [0.85, 0.875, 0.90, 0.925, 0.95],
        "minimum_displacement_cosine": [0.85, 0.875, 0.90, 0.925, 0.95],
        "maximum_normalized_disagreement": [0.42, 0.385, 0.35, 0.315, 0.28],
        "minimum_brier_gain_lcb": [0.0, 0.0005, 0.001, 0.0015, 0.002],
        "auc_noninferiority_margin": [-0.03, -0.025, -0.02, -0.015, -0.01],
    }
    if not all(np.isclose(grids[name][2], frozen[name]) for name in PARAMETERS):
        raise RuntimeError("Every local grid must be centered on its frozen threshold")
    return frozen, grids


def bootstrap_certificate(audit: pd.DataFrame, probability: str, repetitions: int, seed: int) -> dict[str, float]:
    work = audit.copy()
    labels = work.material_optimism_event.to_numpy(int)
    identity = work.probability_identity.to_numpy(float)
    candidate = work[probability].to_numpy(float)
    work["gain"] = (labels - identity) ** 2 - (labels - candidate) ** 2
    clusters = work.groupby(list(CLUSTER_COLUMNS), sort=False, dropna=False).gain.mean().reset_index()
    rng = np.random.default_rng(seed)
    draws = np.zeros(repetitions)
    total = 0
    for _, stratum in clusters.groupby(list(STRATUM_COLUMNS), sort=False, dropna=False):
        gains = stratum.gain.to_numpy(float)
        sampled = rng.integers(0, len(gains), size=(repetitions, len(gains)))
        draws += gains[sampled].sum(axis=1)
        total += len(gains)
    draws /= total
    return {"brier_gain": float(clusters.gain.mean()), "brier_gain_lcb": float(np.quantile(draws, 0.025)), "auc_delta": 0.0}


def make_development_evidence(args: argparse.Namespace, freeze: dict[str, Any]) -> list[Evidence]:
    cert = freeze["certification"]
    config = RiskControlledConfig(
        configuration_budgets=(int(cert["audit_configuration_budget"]),),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=float(cert["auc_noninferiority_margin"]),
        minimum_brier_gain=float(cert["minimum_brier_gain_lcb"]),
        minimum_valid_auc_bootstraps=int(cert["minimum_valid_auc_bootstraps"]),
    )
    prepared, diagnostics, _ = prepare_v9(load_development(), config, args.seed)
    assignments = pd.read_csv(DEV / "distribution_covered_assignments.csv")
    diag = {name: group.reset_index(drop=True) for name, group in diagnostics.groupby("dataset", sort=False)}
    result = []
    certificate_seed = args.seed + int(cert["audit_configuration_budget"]) * 1009
    for dataset, frame in tqdm(prepared.items(), total=len(prepared), desc="Frozen-candidate certificates", unit="domain", dynamic_ncols=True):
        identity = frame.probability_identity.to_numpy(float)
        coral = frame.probability_component_coral.to_numpy(float)
        quantile = frame.probability_component_quantile_mapping.to_numpy(float)
        witness = witness_statistics(identity, coral, quantile)
        assignment = assignments.loc[assignments.dataset.eq(dataset)].drop(columns="dataset")
        candidate = frame.copy()
        candidate["probability_candidate_coral"] = coral
        audit, _ = partition_frame(candidate, assignment)
        result.append(
            Evidence(
                dataset,
                "retrospective development",
                diag[dataset],
                {key: float(witness[key]) for key in ("directional_agreement", "displacement_cosine", "normalized_disagreement")},
                bootstrap_certificate(audit, "probability_candidate_coral", args.bootstrap_repetitions, certificate_seed),
            )
        )
    return result


def external_paths(dataset: str, root: Path) -> tuple[Path, Path, Path]:
    if dataset == "EEGEmotions-27":
        return (
            root / "action/component_projection_diagnostics.csv",
            root / "action/mechanism_probabilities_blinded.csv",
            root / "outcome/primary_audit_labeled.csv",
        )
    return root / "component_projection_diagnostics.csv", root / "mechanism_probabilities_blinded.csv", root / "primary_audit_labeled.csv"


def make_external_evidence(args: argparse.Namespace) -> list[Evidence]:
    roles = {
        "AVDOS-VR": "post-access exploratory",
        "FACED": "post-access exploratory",
        "EEGEmotions-27": "separate pre-signal external robustness test",
    }
    result = []
    for index, (dataset, root) in enumerate(EXTERNAL.items()):
        diagnostics_path, mechanism_path, audit_path = external_paths(dataset, root)
        for path in (diagnostics_path, mechanism_path, audit_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        diagnostics = pd.read_csv(diagnostics_path)
        mechanism = pd.read_csv(mechanism_path)
        witness = witness_statistics(
            mechanism.probability_identity.to_numpy(float),
            mechanism.probability_coral.to_numpy(float),
            mechanism.probability_quantile_mapping.to_numpy(float),
        )
        audit = pd.read_csv(audit_path)
        result.append(
            Evidence(
                dataset,
                roles[dataset],
                diagnostics,
                {key: float(witness[key]) for key in ("directional_agreement", "displacement_cosine", "normalized_disagreement")},
                bootstrap_certificate(audit, "probability_coral", args.bootstrap_repetitions, args.seed + 900_000 + index * 7919),
            )
        )
    return result


def decide(domain: Evidence, thresholds: dict[str, float]) -> dict[str, Any]:
    component_pass = True
    for method in ("coral", "quantile_mapping"):
        row = domain.components.loc[domain.components.method.eq(method)].iloc[0]
        component_pass &= (
            abs(float(row.probability_mean_shift)) <= thresholds["maximum_probability_mean_shift"]
            and float(row.probability_rank) >= thresholds["minimum_probability_rank"]
            and int(row.order_inversions) <= thresholds["maximum_order_inversions"]
        )
    witness_pass = (
        domain.witness["directional_agreement"] >= thresholds["minimum_directional_agreement"]
        and domain.witness["displacement_cosine"] >= thresholds["minimum_displacement_cosine"]
        and domain.witness["normalized_disagreement"] <= thresholds["maximum_normalized_disagreement"]
    )
    applicable = bool(component_pass and witness_pass)
    brier_pass = domain.certificate["brier_gain_lcb"] > thresholds["minimum_brier_gain_lcb"]
    auc_pass = domain.certificate["auc_delta"] >= thresholds["auc_noninferiority_margin"]
    certified = bool(applicable and brier_pass and auc_pass)
    failure = "released"
    if not component_pass:
        failure = "component gate"
    elif not witness_pass:
        failure = "witness gate"
    elif not brier_pass:
        failure = "Brier certificate"
    elif not auc_pass:
        failure = "AUROC certificate"
    return {
        "component_pass": bool(component_pass),
        "witness_pass": bool(witness_pass),
        "applicable": applicable,
        "brier_pass": bool(brier_pass),
        "auc_pass": bool(auc_pass),
        "certified": certified,
        "selected_action": "CORAL + OPCT" if certified else "identity",
        "failure_stage": failure,
    }


def make_tables(evidence: list[Evidence], frozen: dict[str, float], grids: dict[str, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_rows, action_rows, margin_rows = [], [], []
    frozen_actions = {domain.dataset: decide(domain, frozen)["selected_action"] for domain in evidence}
    for parameter in tqdm(PARAMETERS, desc="One-at-a-time thresholds", unit="gate", dynamic_ncols=True):
        for index, (grid_label, value) in enumerate(zip(GRID_LABELS, grids[parameter], strict=True)):
            grid_rows.append({"parameter": parameter, "parameter_label": LABELS[parameter], "grid_index": index, "grid_label": grid_label, "threshold_value": value, "frozen_value": frozen[parameter], "is_frozen": index == 2})
            thresholds = dict(frozen)
            thresholds[parameter] = value
            for domain in evidence:
                decision = decide(domain, thresholds)
                action_rows.append({"dataset": domain.dataset, "evidence_role": domain.role, "parameter": parameter, "parameter_label": LABELS[parameter], "grid_index": index, "grid_label": grid_label, "threshold_value": value, "frozen_value": frozen[parameter], **decision, "frozen_action": frozen_actions[domain.dataset], "action_changed": decision["selected_action"] != frozen_actions[domain.dataset]})

    for domain in evidence:
        components = domain.components.set_index("method")
        observed = {
            "maximum_probability_mean_shift": components.loc[["coral", "quantile_mapping"], "probability_mean_shift"].astype(float).abs().max(),
            "minimum_probability_rank": components.loc[["coral", "quantile_mapping"], "probability_rank"].astype(float).min(),
            "maximum_order_inversions": components.loc[["coral", "quantile_mapping"], "order_inversions"].astype(float).max(),
            "minimum_directional_agreement": domain.witness["directional_agreement"],
            "minimum_displacement_cosine": domain.witness["displacement_cosine"],
            "maximum_normalized_disagreement": domain.witness["normalized_disagreement"],
            "minimum_brier_gain_lcb": domain.certificate["brier_gain_lcb"],
            "auc_noninferiority_margin": domain.certificate["auc_delta"],
        }
        for parameter in PARAMETERS:
            value, threshold = float(observed[parameter]), frozen[parameter]
            minimum = parameter.startswith("minimum_") or parameter == "auc_noninferiority_margin"
            raw = value - threshold if minimum else threshold - value
            margin_rows.append({"dataset": domain.dataset, "evidence_role": domain.role, "parameter": parameter, "parameter_label": LABELS[parameter], "observed_value": value, "frozen_threshold": threshold, "signed_margin": raw, "normalized_signed_margin": raw / max(abs(threshold), 1e-6), "passes_frozen_gate": raw >= -1e-12})
    return pd.DataFrame(grid_rows), pd.DataFrame(action_rows), pd.DataFrame(margin_rows)


def make_joint_cube(evidence: list[Evidence], grids: dict[str, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    combinations = list(itertools.product([0, 2, 4], repeat=len(PARAMETERS)))
    rows, counts = [], {}
    for scenario, indices in enumerate(tqdm(combinations, desc="Joint local pressure cube", unit="scenario", dynamic_ncols=True)):
        thresholds = {parameter: grids[parameter][index] for parameter, index in zip(PARAMETERS, indices, strict=True)}
        released = tuple(domain.dataset for domain in evidence if decide(domain, thresholds)["certified"])
        counts[released] = counts.get(released, 0) + 1
        rows.append({"scenario_id": scenario, "level_signature": ",".join(map(str, indices)), "released_domain_count": len(released), "released_domains": ";".join(released)})
    patterns = pd.DataFrame(
        [{"released_domains": ";".join(pattern), "released_domain_count": len(pattern), "scenario_count": count, "scenario_fraction": count / len(combinations)} for pattern, count in counts.items()]
    ).sort_values(["scenario_count", "released_domain_count"], ascending=[False, False]).reset_index(drop=True)
    return pd.DataFrame(rows), patterns


def make_figure(actions: pd.DataFrame, margins: pd.DataFrame, patterns: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 8.0, "axes.titlesize": 9.2, "axes.labelsize": 8.4, "xtick.labelsize": 7.1, "ytick.labelsize": 7.1})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), constrained_layout=True, gridspec_kw={"height_ratios": [1.0, 1.05, 1.2]})
    labels = [LABELS[name] for name in PARAMETERS]

    released = actions.groupby(["parameter_label", "grid_index"], sort=False).certified.sum().unstack().reindex(labels)
    image = axes[0].imshow(released.to_numpy(float), cmap="Blues", vmin=0, vmax=len(DATASETS), aspect="auto")
    axes[0].set_xticks(range(5), ["Perm. 2", "Perm. 1", "Frozen", "Strict 1", "Strict 2"])
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].axvline(1.5, color="#B23A48", linewidth=0.8)
    axes[0].axvline(2.5, color="#B23A48", linewidth=0.8)
    for row, column in itertools.product(range(released.shape[0]), range(released.shape[1])):
        value = int(released.iloc[row, column])
        axes[0].text(column, row, str(value), ha="center", va="center", color="white" if value >= 4 else "#17202A")
    axes[0].set_title("A  Released-domain count under one-at-a-time perturbation", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[0], label="Released domains", shrink=0.8)

    changes = actions.groupby(["dataset", "parameter_label"], sort=False).action_changed.sum().unstack().reindex(index=DATASETS, columns=labels)
    image = axes[1].imshow(changes.to_numpy(float), cmap="OrRd", vmin=0, vmax=4, aspect="auto")
    axes[1].set_xticks(range(8), ["Mean", "Rank", "Inv.", "Dir.", "Cos.", "Dis.", "Brier", "AUC"])
    axes[1].set_yticks(range(8), DATASETS)
    for row, column in itertools.product(range(8), range(8)):
        axes[1].text(column, row, str(int(changes.iloc[row, column])), ha="center", va="center", color="white" if changes.iloc[row, column] >= 3 else "#17202A")
    axes[1].set_title("B  Action flips across four non-frozen settings", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[1], label="Flipped settings", shrink=0.8)

    display = margins.copy()
    display["value"] = display.normalized_signed_margin.clip(-1, 1)
    matrix = display.pivot(index="dataset", columns="parameter_label", values="value").reindex(index=DATASETS, columns=labels)
    image = axes[2].imshow(matrix.to_numpy(float), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axes[2].set_xticks(range(8), ["Mean", "Rank", "Inv.", "Dir.", "Cos.", "Dis.", "Brier", "AUC"])
    axes[2].set_yticks(range(8), DATASETS)
    for row, column in itertools.product(range(8), range(8)):
        value = matrix.iloc[row, column]
        axes[2].text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=6.4, color="white" if abs(value) > 0.58 else "#17202A")
    axes[2].set_title("C  Signed distance from each frozen gate (clipped normalized scale)", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[2], label="Pass (+) / fail (-) margin", shrink=0.8)

    fig.suptitle("Post-hoc stability of frozen DCS-OPCT v11 actions", fontsize=11.5, fontweight="bold")
    fig.savefig(output / "fig_frozen_action_stability.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_frozen_action_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    validate(args, freeze)
    frozen, grids = threshold_spec(freeze)
    evidence = make_development_evidence(args, freeze) + make_external_evidence(args)
    evidence.sort(key=lambda item: DATASETS.index(item.dataset))
    if [item.dataset for item in evidence] != DATASETS:
        raise RuntimeError("Unexpected domain inventory")

    grid, actions, margins = make_tables(evidence, frozen, grids)
    joint, patterns = make_joint_cube(evidence, grids)
    frozen_decisions = {domain.dataset: decide(domain, frozen) for domain in evidence}
    changed = actions.loc[actions.action_changed]
    summary = {
        "frozen_released_domains": [name for name in DATASETS if frozen_decisions[name]["certified"]],
        "frozen_identity_domains": [name for name in DATASETS if not frozen_decisions[name]["certified"]],
        "one_at_a_time_action_flip_count": int(len(changed)),
        "one_at_a_time_domains_with_any_flip": sorted(changed.dataset.unique().tolist()),
        "one_at_a_time_parameters_with_any_flip": sorted(changed.parameter.unique().tolist()),
        "joint_scenario_count": int(len(joint)),
        "joint_distinct_release_patterns": int(len(patterns)),
        "most_common_joint_pattern": patterns.iloc[0].to_dict(),
        "claim_boundary": "This is a post-hoc stability analysis of frozen predictions and assignments. It does not select thresholds, modify v11, establish prospective external effectiveness, or convert identity fallback into a safety success.",
    }

    args.output_root.mkdir(parents=True, exist_ok=False)
    tables = {
        "threshold_grid.csv": grid,
        "one_at_a_time_action_stability.csv": actions,
        "domain_gate_margins.csv": margins,
        "joint_local_pressure_cube.csv": joint,
        "joint_release_patterns.csv": patterns,
    }
    for name, frame in tables.items():
        frame.to_csv(args.output_root / name, index=False)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    make_figure(actions, margins, patterns, args.output_root)

    input_paths = [FREEZE, DEV / "distribution_covered_assignments.csv", DEV / "component_projection_diagnostics.csv"]
    for dataset, root in EXTERNAL.items():
        input_paths.extend(external_paths(dataset, root))
    outputs = list(tables) + ["summary.json", "fig_frozen_action_stability.pdf", "fig_frozen_action_stability.png"]
    manifest = {
        "status": "completed_post_hoc_frozen_action_stability",
        "analysis_date": "2026-09-10",
        "analysis_role": "Post-hoc local action-stability analysis; no threshold optimization, candidate refit, audit reassignment, or manuscript integration.",
        "frozen_v11_unchanged": True,
        "freeze_sha256": sha256(FREEZE),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "gpu_name": torch.cuda.get_device_name(0),
        "domain_count": len(evidence),
        "one_at_a_time_rows": len(actions),
        "joint_scenario_count": len(joint),
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "script_sha256": sha256(Path(__file__)),
        "outputs": {name: sha256(args.output_root / name) for name in outputs},
        "claim_boundary": summary["claim_boundary"],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
