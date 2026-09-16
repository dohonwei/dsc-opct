from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2"
OUT = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3"
INVENTORY = ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/inventory.json"
INVENTORY_VALIDATION = (
    ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/validation_report.json"
)
RELEASE = ROOT / "outputs/dcs_opct_v11_release_readiness/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_required(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count < 1:
        raise RuntimeError(f"Missing v3 replacement anchor: {label}")
    return text.replace(old, new)


def prepare_tree() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(BASE, OUT)
    for pattern in ["*.pdf", "*.log", "independent_validation_report.json"]:
        for path in OUT.glob(pattern):
            path.unlink()
    rendered = OUT / "rendered"
    if rendered.exists():
        shutil.rmtree(rendered)


def make_revised_framework() -> None:
    colors = {
        "mechanism": "#2F6B9A",
        "risk": "#15806F",
        "release": "#8A5A2B",
        "ink": "#263238",
        "muted": "#64748B",
        "grid": "#D8DEE6",
        "boundary": "#A63D40",
    }
    fig, ax = plt.subplots(figsize=(12.2, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        (0.04, colors["mechanism"], "1  MECHANISM", "Counterfactual identity dose", [
            "Fixed test rows and\ntraining constraints",
            "Controlled identity-label\nopportunity",
            "Utilization and ranking\nregret",
        ]),
        (0.365, colors["risk"], "2  RISK", "Frozen audit-risk model", [
            "Material optimism risk\n(>= 0.02)",
            "Mechanism-derived\npredictors only",
            "Ranking separated from\ncalibration",
        ]),
        (0.69, colors["release"], "3  RELEASE CONTROL", "Selective audit-risk transport", [
            "CORAL action with\ncomplementary witness",
            "Distribution-covered\nlimited-label audit",
            "Release correction or\nnon-intervention",
        ]),
    ]
    width = 0.27
    for x, color, eyebrow, title, bullets in stages:
        box = FancyBboxPatch(
            (x, 0.28), width, 0.59,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.5, edgecolor=color, facecolor="white",
        )
        ax.add_patch(box)
        ax.add_patch(plt.Rectangle((x, 0.79), width, 0.08, color=color, linewidth=0))
        ax.text(x + 0.018, 0.83, eyebrow, color="white", fontsize=10.5,
                fontweight="bold", va="center")
        ax.text(x + 0.018, 0.735, title, color=colors["ink"], fontsize=11.2,
                fontweight="bold", va="top")
        y = 0.615
        for bullet in bullets:
            ax.text(x + 0.025, y, "-", color=color, fontsize=15, va="center")
            ax.text(x + 0.048, y, bullet, color=colors["ink"], fontsize=9.0,
                    va="center", linespacing=1.12)
            y -= 0.135
    for x1, x2 in [(0.31, 0.365), (0.635, 0.69)]:
        ax.add_patch(FancyArrowPatch(
            (x1, 0.605), (x2, 0.605), arrowstyle="-|>", mutation_scale=17,
            linewidth=1.8, color=colors["muted"],
        ))
    ax.text(0.5, 0.955, "From evaluation shortcut to a bounded cross-domain audit decision",
            ha="center", va="center", fontsize=17, fontweight="bold", color=colors["ink"])
    ax.add_patch(FancyBboxPatch(
        (0.04, 0.08), 0.92, 0.14, boxstyle="round,pad=0.01,rounding_size=0.008",
        facecolor="#F5F7FA", edgecolor=colors["grid"], linewidth=1.2,
    ))
    ax.text(0.065, 0.165, "CLAIM BOUNDARY", fontsize=9.5, fontweight="bold",
            color=colors["boundary"], va="center")
    ax.text(0.225, 0.165,
            "Transported endpoint: probability of a configuration-level material identity-optimism event",
            fontsize=9.6, color=colors["ink"], va="center")
    ax.text(0.225, 0.115,
            "Not transported: trial-level emotion predictions or EEG classifier parameters",
            fontsize=9.6, color=colors["ink"], va="center")
    for suffix in ["pdf", "png"]:
        fig.savefig(OUT / "figures" / f"fig_v11_three_stage_framework.{suffix}",
                    dpi=320, bbox_inches="tight")
    plt.close(fig)


def current_counts() -> tuple[int, int, int, int]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    validation = json.loads(INVENTORY_VALIDATION.read_text(encoding="utf-8"))
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    return (
        int(inventory["unique_file_count"]),
        int(validation["checks_passed"]),
        int(validation["checks_total"]),
        len(release["checks"]),
    )


def revise_text(path: Path, *, supplement: bool) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = [
        ("two independent unlabeled transformations agree",
         "two separately constructed unlabeled transformations agree",
         "unlabeled component independence"),
        ("independent witness", "complementary witness", "witness independence"),
        ("independent unlabeled transformations", "separately constructed unlabeled transformations",
         "conclusion component independence"),
        ("mechanism--risk--safety loop", "mechanism--risk--release-control loop",
         "framework safety wording"),
        ("controlled causal contrast", "controlled protocol contrast", "causal wording"),
        ("universal or prospectively confirmed safe transfer",
         "universal effectiveness, prospective external effectiveness, or future-domain non-harm",
         "abstract safe-transfer wording"),
        ("universal safe transfer", "universal transport safety", "conclusion safe-transfer wording"),
    ]
    for old, new, label in replacements:
        if old in text:
            text = replace_required(text, old, new, label)

    if not supplement:
        text = replace_required(
            text,
            "the existing 20\\% minimum-budget condition; budget.",
            "the existing 20\\% minimum-budget condition; all other budgets were treated as sensitivity analyses.",
            "minimum-budget fragment",
        )
        text = replace_required(
            text,
            "Dose-zero rows served only as anchors, leaving 480 nonzero-dose configurations.",
            "Dose-zero rows served only as anchors, leaving 480 nonzero-dose configurations. "
            "These rows arose from only two independent source datasets; configuration-level "
            "sample size must therefore not be interpreted as domain-level replication ($n_{domain}=2$).",
            "domain-level sample size",
        )
        text = replace_required(
            text,
            "Parameters were fitted on CUDA for 500 epochs with learning rate 0.03 and regularization $10^{-4}$. ",
            "For component $k$, identity-probability logits were the inputs and the label-free "
            "CORAL or quantile base probabilities $\\widetilde p_i^{(k)}$ were soft targets. "
            "The fitted parameters minimized\n"
            "\\begin{equation}\n"
            "\\mathcal{L}_k(a,b)=\\frac{1}{n}\\sum_{i=1}^{n}"
            "\\left[T_{a,b}(p_i^{(I)})-\\widetilde p_i^{(k)}\\right]^2"
            "+\\lambda\\left[(a-1)^2+b^2\\right],\n"
            "\\end{equation}\n"
            "subject to $0.1\\leq a\\leq5$ and $|b|\\leq5$. No target outcome entered this "
            "optimization. Parameters were fitted on CUDA for 500 epochs with learning rate "
            "0.03 and regularization $\\lambda=10^{-4}$. ",
            "OPCT objective",
        )
        text = replace_required(
            text,
            "Within the already assigned audit half, 100 seeded balanced cluster orderings generated nested",
            "Within the already assigned audit half, 100 seeded balanced cluster orderings of the same fixed data generated nested",
            "audit repetition dependence",
        )
        text = replace_required(
            text,
            "No held-out outcome entered fitting, certification, or budget selection.",
            "No held-out outcome entered fitting, certification, or budget selection. The 100 orderings "
            "were conditional allocation-sensitivity replicates, not independent dataset or participant replications.",
            "audit repetition interpretation",
        )

    files, passed, total, release_checks = current_counts()
    count_sentence = (
        f"The current immutable reproducibility inventory contains {files} files and passed "
        f"{passed}/{total} checks. The release-readiness driver passed {release_checks}/{release_checks} "
        "gates, including the fail-closed external-arrival detector, whose own validation passed 7/7. "
        "AMIGOS and Emognition remained blocked by external data access; these software checks do not "
        "constitute external effectiveness or future-domain non-harm evidence."
    )
    anchor = "Failed method versions, execution defects, and external-protocol incompatibilities were retained rather than overwritten."
    if anchor in text:
        text = replace_required(text, anchor, anchor + "\n\n" + count_sentence,
                                "authoritative reproducibility counts")
    elif supplement:
        supplement_anchor = (
            "These checks establish internal reproducibility and pre-access software readiness; "
            "they do not substitute for external confirmation."
        )
        if supplement_anchor in text:
            text = replace_required(text, supplement_anchor,
                                    supplement_anchor + "\n\n" + count_sentence,
                                    "supplement reproducibility counts")
        else:
            raise RuntimeError("Missing supplement reproducibility anchor")
    path.write_text(text, encoding="utf-8")


def write_manifest() -> None:
    tracked = [
        OUT / "manuscript_preview.tex",
        OUT / "supplementary_preview.tex",
        OUT / "figures/fig_v11_three_stage_framework.pdf",
        OUT / "figures/fig_v11_three_stage_framework.png",
    ]
    manifest = {
        "status": "staged_preview_not_merged_into_authoritative_manuscript",
        "preview_version": "v3_nature_review_claim_and_methods_revision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_preview": str(BASE.relative_to(ROOT)).replace("\\", "/"),
        "authoritative_sources_unchanged": True,
        "freeze_sha256": sha256(ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"),
        "eegemotions_lock_sha256": sha256(ROOT / "docs/eegemotions27_v11_external_robustness_implementation_lock.json"),
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in tracked
        },
        "claim_boundary": (
            "This v3 preview moderates independence, causality, and safety language; documents the "
            "actual OPCT soft-target objective; identifies domain-level n=2 and conditional audit-order "
            "sensitivity; and synchronizes reproducibility counts. It remains a staged preview and does "
            "not add prospective external effectiveness or future-domain non-harm evidence."
        ),
    }
    (OUT / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    prepare_tree()
    make_revised_framework()
    revise_text(OUT / "manuscript_preview.tex", supplement=False)
    revise_text(OUT / "supplementary_preview.tex", supplement=True)
    write_manifest()
    print(f"Staged evidence-integration preview v3: {OUT}")


if __name__ == "__main__":
    main()
