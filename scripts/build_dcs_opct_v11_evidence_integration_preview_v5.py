from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4"
HARM = ROOT / "outputs/dcs_opct_v11_endpoint_harmonization_20260910"
OUT = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v5"
INVENTORY = ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/inventory.json"
INVENTORY_VALIDATION = ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/validation_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def sync_counts(text: str) -> str:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    validation = json.loads(INVENTORY_VALIDATION.read_text(encoding="utf-8"))
    replacement = (f"The current immutable reproducibility inventory contains {inventory['unique_file_count']} files "
                   f"and passed {validation['checks_passed']}/{validation['checks_total']} checks.")
    return re.sub(r"The current immutable reproducibility inventory contains \d+ files and passed \d+/\d+ checks\.",
                  replacement, text)


def prepare_tree() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(BASE, OUT)
    for pattern in ("*.pdf", "*.log", "independent_validation_report.json"):
        for path in OUT.glob(pattern):
            path.unlink()
    rendered = OUT / "rendered"
    if rendered.exists():
        shutil.rmtree(rendered)
    shutil.copy2(HARM / "table_endpoint_harmonization.tex", OUT / "tables/table_endpoint_harmonization.tex")


def build_manuscript() -> Path:
    text = (BASE / "manuscript_preview.tex").read_text(encoding="utf-8")
    anchor = ("The registered EEGEmotions-27 task used fixed stimulus-category polarity: emotion-video IDs 4, 20, 22, and 25 were positive, whereas IDs 5, 6, 13, 17, 18, and 24 were negative. These labels describe the assigned stimulus category and are not participant-experienced valence or arousal. One feature row represented one participant--physical-video pair after within-trial aggregation. The fixed grid used the 14-channel Emotiv X montage, three representations, two classifiers, five split seeds, five subject folds, four physical-stimulus folds, and five identity doses. Both participant and physical-video identities were held out in the dual-unseen endpoint. All GPU MLP fits used the frozen 40-epoch CUDA implementation, and the action and distribution-covered audit assignment were hashed before category-polarity outcomes entered the release decision.")
    addition = anchor + ("\n\nThe dataset-specific emotion endpoints were not treated as one interchangeable psychological construct. "
        "DEAP, MAHNOB-HCI, EPPVR, CASE, CEAP, and DREAMER used participant-rated dimensional affect under "
        "dataset-specific cut points; SEED-IV, AVDOS-VR, FACED, EEGEmotions-27, and the reserved AMIGOS protocol "
        "used fixed stimulus or segment-category contrasts. Emognition reserves participant SAM ratings but excludes "
        "the midpoint. These labels define each counterfactual experiment locally. The quantity transferred by "
        "DCS-OPCT is instead the common configuration-level event $Y_{c,d}$, conditional on that local task definition. "
        "No trial-level emotion labels, scores, or classifier outputs were pooled across datasets; the complete endpoint "
        "ontology and exclusions are reported in Supplementary Table~S1.")
    text = replace_once(text, anchor, addition, "endpoint harmonization methods")
    limitation = ("Finally, neither external confirmation attempt completed a successful registered effectiveness test.")
    replacement = ("The EPPVR raw and processing chain verified 30 participants, 14 trials per participant, ten "
        "synchronized channels, 100-Hz sampling, FP1/FP2 as channels 0/1, and a 10-s per-trial baseline followed by "
        "60 s of post-baseline data. However, the local archive and code do not establish the acquisition-device model, "
        "reference and ground, channels 5--9, the third label column, ethics committee and approval number, written "
        "informed consent, or privacy and secondary-analysis authorization. These items remain author-confirmation "
        "requirements and must be resolved before submission; none is inferred from array shape or preprocessing code.\n\n" + limitation)
    text = replace_once(text, limitation, replacement, "EPPVR governance limitation")
    text = sync_counts(text)
    path = OUT / "manuscript_preview.tex"
    path.write_text(text, encoding="utf-8")
    return path


def build_supplement() -> Path:
    text = (BASE / "supplementary_preview.tex").read_text(encoding="utf-8")
    text = replace_once(text, "\\usepackage{array}\n", "\\usepackage{array}\n\\usepackage{longtable}\n",
                        "longtable package")
    text = replace_once(text, "\\begin{document}\n", "\\begin{document}\n\\renewcommand{\\thetable}{S\\arabic{table}}\n",
                        "supplementary table numbering")
    anchor = "\\section{Counterfactual identity-dose contract}\n"
    section = r"""\section{Endpoint ontology and EPPVR acquisition trace}

The task labels used to construct the counterfactual experiments are heterogeneous by design. Participant-experienced dimensional ratings were used in DEAP, MAHNOB-HCI, EPPVR, CASE, CEAP, and DREAMER, with fixed but dataset-specific cut points. SEED-IV used prespecified extreme stimulus-category contrasts; AVDOS-VR used the official segment-category arousal mapping; FACED and EEGEmotions-27 used stimulus-category polarity; and the reserved AMIGOS protocol uses intended stimulus quadrants. Emognition reserves participant-experienced SAM valence and arousal with ratings 6--9 high, 1--4 low, and midpoint 5 excluded. These ontologies are not declared equivalent. They are connected only through the higher-level operational endpoint: whether the local identity-dose experiment produces at least 0.02 balanced-accuracy amplification.

\begingroup
\footnotesize
\setlength{\tabcolsep}{3pt}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\renewcommand{\arraystretch}{1.08}
\input{tables/table_endpoint_harmonization.tex}
\endgroup

The local EPPVR source contains 30 byte-matched participant files in both the project dataset directory and the archived handover directory. Each file contains 14 trials with arrays shaped $14\times10\times7000$ and labels shaped $14\times3$. The processing chain identifies channels 0 and 1 as FP1 and FP2, channels 2--4 as GSR/EDA, PPG/BVP, and SKT/HST, and uses 100 Hz. The first 1,000 samples form a 10-s baseline and the remaining 6,000 samples form 60 s of post-baseline recording. Only valence and arousal label columns enter the executed analysis, with scores at least 5 assigned to the high class. The third label column and channels 5--9 are not documented by the available local materials.

This trace is deliberately fail-closed. Device manufacturer/model, electrode reference and ground, the remaining channel placements, ethics committee, approval identifier, written informed consent, and privacy/secondary-analysis authorization are marked \texttt{AUTHOR\_CONFIRMATION\_REQUIRED}. The signal contract does not prove these governance facts. EPPVR also has one session per participant and no verified cross-participant physical-stimulus identifier, so only the subject axis is admissible and trial position cannot be repurposed as shared stimulus identity.

"""
    text = replace_once(text, anchor, section + anchor, "endpoint harmonization supplement")
    text = sync_counts(text)
    path = OUT / "supplementary_preview.tex"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    report = json.loads((HARM / "independent_validation_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("checks_passed") != report.get("checks_total"):
        raise RuntimeError("Endpoint harmonization source has not passed independent validation")
    prepare_tree()
    manuscript, supplement = build_manuscript(), build_supplement()
    tracked = [manuscript, supplement, OUT / "tables/table_endpoint_harmonization.tex"]
    manifest = {"status": "staged_preview_not_merged_into_authoritative_manuscript",
        "preview_version": "v5_endpoint_harmonization_and_eppvr_trace",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_preview": str(BASE.relative_to(ROOT)).replace("\\", "/"),
        "authoritative_sources_unchanged": True,
        "freeze_sha256": sha256(ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"),
        "endpoint_harmonization_manifest_sha256": sha256(HARM / "manifest.json"),
        "files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked},
        "claim_boundary": "Endpoint heterogeneity is explicit; EPPVR governance gaps remain author-confirmation requirements."}
    (OUT / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Staged evidence-integration preview v5: {OUT}")


if __name__ == "__main__":
    main()
