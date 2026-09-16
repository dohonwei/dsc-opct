from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_endpoint_harmonization_20260910"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
EPPVR_ROOT = Path(r"E:\AA发表论文的数据\dataset\EPPVR")
EPPVR_ARCHIVE = Path(r"H:\MCC\万春霆交接资料汇总\情绪识别程序\data_hmd")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_row(dataset: str, task: str, construct: str, rule: str, ontology: str,
             exclusions: str, axis: str, role: str, boundary: str,
             provenance: str, status: str = "VERIFIED_LOCAL_ARTIFACT") -> dict[str, str]:
    return {
        "dataset": dataset, "task": task, "label_construct": construct,
        "mapping_rule": rule, "ontology": ontology, "exclusions": exclusions,
        "validated_axis": axis, "evidence_role": role,
        "comparability_boundary": boundary, "provenance_paths": provenance,
        "verification_status": status,
    }


def endpoint_rows() -> list[dict[str, str]]:
    rated = ("Participant-reported dimensional affect; dataset-specific scales and cut points "
             "do not make scores metrically interchangeable.")
    p = lambda *items: "; ".join(items)
    rows: list[dict[str, str]] = []
    for dataset in ("DEAP", "MAHNOB-HCI"):
        provenance = p("scripts/build_unified_frontal_features.py",
                       "scripts/run_counterfactual_identity_dose.py",
                       "outputs/counterfactual_identity_dose_crossed_valid/run_manifest.json")
        for task in ("arousal", "valence"):
            rows.append(make_row(dataset, task, f"experienced {task}",
                "score >= 5 is high; score < 5 is low", "participant self-report", "none",
                "subject and verified physical stimulus", "Stage-I mechanism and Stage-II source",
                rated, provenance))
    for task in ("arousal", "valence"):
        rows.append(make_row("EPPVR", task, f"experienced {task}",
            "score >= 5 is high; score < 5 is low", "participant self-report", "none",
            "subject only; record position is not a verified shared stimulus ID",
            "retrospective development", rated + " Raw label column 3 is undocumented locally.",
            p("scripts/build_unified_frontal_features.py",
              "scripts/run_eppvr_subject_shortcut_validation.py",
              "outputs/eppvr_subject_shortcut_external_validation/run_manifest.json")))
    for dataset, provenance in (
        ("CASE", p("scripts/build_case_trial_features.py", "scripts/run_case_counterfactual_identity_dose.py",
                   "outputs/case_counterfactual_identity_dose/run_manifest.json")),
        ("CEAP-360VR", p("scripts/build_ceap_trial_features.py", "scripts/run_ceap_counterfactual_identity_dose.py",
                         "outputs/ceap_counterfactual_identity_dose/run_manifest.json")),
    ):
        for task in ("arousal", "valence"):
            rows.append(make_row(dataset, task, f"experienced {task}",
                "score >= 5 is high; score < 5 is low", "participant annotation/self-report",
                "registered trial eligibility only", "subject and physical stimulus",
                "retrospective development", rated, provenance))
    rows.extend([
        make_row("SEED-IV", "valence", "stimulus emotion extreme contrast", "happy vs sad",
            "fixed stimulus category", "neutral and fear excluded", "subject only in executed dose analysis",
            "retrospective development after earlier negative test",
            "Category contrast is not participant-experienced valence.",
            p("scripts/build_seediv_trial_features.py", "outputs/seediv_trial_features/build_manifest.json",
              "outputs/seediv_valence_counterfactual_external/run_manifest.json")),
        make_row("SEED-IV", "arousal", "stimulus emotion extreme contrast", "fear vs neutral",
            "fixed stimulus category", "happy and sad excluded", "subject only in executed dose analysis",
            "retrospective development after earlier negative test",
            "Category contrast is not participant-experienced arousal.",
            p("scripts/build_seediv_trial_features.py", "outputs/seediv_trial_features/build_manifest.json",
              "outputs/seediv_arousal_counterfactual_external/run_manifest.json")),
    ])
    for task in ("arousal", "valence"):
        rows.append(make_row("DREAMER", task, f"experienced {task}",
            "rating > 3 is high; rating <= 3 is low", "participant self-report",
            "none in primary; sensitivity excludes rating 3", "subject only",
            "retrospective development", rated,
            p("scripts/build_dreamer_trial_features.py", "outputs/dreamer_trial_features/build_manifest.json",
              "outputs/dreamer_counterfactual_external/run_manifest.json")))
    rows.extend([
        make_row("AVDOS-VR", "arousal", "official segment arousal category",
            "Neutral=0; Negative=1; Positive=1", "fixed segment category",
            "valence, HRV, and Mean_BPM excluded", "subject only",
            "post-access exploratory PPG stress test",
            "Segment-category arousal is not a 1-9 participant-experienced rating.",
            p("docs/avdos_v11_effective_protocol_status.md", "docs/avdos_v11_label_semantics_audit.json",
              "outputs/avdos_exploratory_ppg_dose/run_manifest.json")),
        make_row("FACED", "category_polarity", "stimulus-category affective polarity",
            "amusement/inspiration/joy/tenderness vs anger/disgust/fear/sadness",
            "fixed stimulus category", "neutral videos 13-16 excluded",
            "subject; stimulus defines crossed cells only", "post-access exploratory common-montage stress test",
            "Stimulus polarity is not participant-experienced valence or arousal.",
            p("scripts/faced_v11_contract.py", "docs/faced_v11_post_access_common_montage_protocol_amendment_001.json",
              "outputs/faced_v11_post_access_common_montage_features/build_manifest.json")),
        make_row("EEGEmotions-27", "category_polarity", "category-derived affective polarity",
            "4 prototypical positive IDs vs 6 prototypical negative IDs",
            "fixed stimulus category with strong self-reported elicitation",
            "17 mixed/context-sensitive categories excluded", "subject and physical emotion/video ID",
            "separate pre-signal external robustness test",
            "Category polarity is not participant-experienced valence or arousal; identity fallback is non-intervention.",
            p("docs/eegemotions27_v11_category_polarity_mapping.md",
              "docs/eegemotions27_v11_external_robustness_implementation_lock.json",
              "outputs/eegemotions27_v11_external_robustness/independent_validation_report.json")),
    ])
    for task in ("valence", "arousal"):
        rows.append(make_row("AMIGOS", task, f"expected stimulus {task}",
            "published 8/8 high-low short-video quadrants", "fixed intended stimulus quadrant",
            "group/long-video branch excluded; complete 16-video subjects required",
            "prospectively reserved subject and physical stimulus", "BLOCKED_EXTERNAL_ACCESS protocol only",
            "Intended stimulus quadrant is not participant-experienced affect; no participant result exists.",
            p("docs/amigos_v11_external_confirmation_implementation_protocol.md", "scripts/amigos_v11_contract.py"),
            "PROTOCOL_ONLY_NO_PARTICIPANT_VALUES"))
    for task in ("valence", "arousal"):
        rows.append(make_row("Emognition", task, f"experienced {task}",
            "SAM 6-9 high; 1-4 low; 5 excluded", "participant self-report",
            "midpoint 5, baseline, washout, questionnaire, and missing task ratings excluded",
            "prospectively reserved subject and physical stimulus", "BLOCKED_EXTERNAL_ACCESS protocol only",
            "Construct name aligns with dimensional affect, but its scale/exclusion rule is dataset specific; no result exists.",
            p("docs/emognition_v11_external_confirmation_protocol.md", "scripts/emognition_v11_contract.py"),
            "PROTOCOL_ONLY_NO_PARTICIPANT_VALUES"))
    return rows


def inspect_eppvr() -> dict[str, object]:
    files = sorted(EPPVR_ROOT.glob("s*.dat"))
    archived = sorted(EPPVR_ARCHIVE.glob("s*.dat"))
    if len(files) != 30 or len(archived) != 30:
        raise RuntimeError("Expected 30 EPPVR files in both locations")
    data_shapes, label_shapes, hashes = set(), set(), {}
    archive_identical = True
    for path in tqdm(files, desc="Audit EPPVR raw contract", unit="subject", dynamic_ncols=True):
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        if set(payload) != {"data", "label"}:
            raise RuntimeError(f"Unexpected EPPVR keys: {path.name}")
        data_shapes.add(tuple(np.asarray(payload["data"]).shape))
        label_shapes.add(tuple(np.asarray(payload["label"]).shape))
        hashes[path.name] = sha256(path)
        archive_identical &= hashes[path.name] == sha256(EPPVR_ARCHIVE / path.name)
    unknown = "AUTHOR_CONFIRMATION_REQUIRED"
    return {
        "dataset": "EPPVR", "inspection_status": "LOCAL_RAW_AND_PROCESSING_CHAIN_VERIFIED",
        "participants": 30, "trials_per_participant": 14,
        "raw_data_shape": sorted(map(list, data_shapes)), "raw_label_shape": sorted(map(list, label_shapes)),
        "raw_keys": ["data", "label"], "synchronized_channels": 10,
        "locally_documented_channel_order": {"0": "FP1", "1": "FP2", "2": "GSR/EDA",
            "3": "PPG/BVP", "4": "SKT/HST", "5-9": unknown},
        "sampling_rate_hz": 100,
        "sampling_rate_evidence": [
            "H:/MCC/万春霆交接资料汇总/情绪识别程序/main_hmd.py",
            "H:/MCC/万春霆交接资料汇总/情绪识别程序/feature_extract_hmd.py",
            "scripts/build_unified_frontal_features.py"],
        "trial_timing": {"total_samples": 7000, "baseline_samples": 1000,
            "baseline_seconds": 10.0, "post_baseline_samples": 6000,
            "post_baseline_seconds": 60.0, "session_count_per_participant": 1},
        "label_columns": {"0": "valence self-rating", "1": "arousal self-rating", "2": unknown},
        "executed_binary_rule": "score >= 5 is high; score < 5 is low",
        "shared_physical_stimulus_id": "UNVERIFIED; ordinal trial position is forbidden as shared identity",
        "baseline_availability": "verified per trial from first 1000 samples",
        "device_manufacturer_model": unknown, "electrode_reference_and_ground": unknown,
        "sensor_placement_for_channels_5_to_9": unknown, "ethics_committee": unknown,
        "ethics_approval_number": unknown, "written_informed_consent": unknown,
        "privacy_and_secondary_analysis_authorization": unknown,
        "archive_copy_byte_identity": archive_identical, "source_subject_sha256": hashes,
        "interpretation_boundary": "Local files establish signal structure, not device identity or research governance.",
    }


def tex_escape(value: str) -> str:
    for old, new in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        value = value.replace(old, new)
    value = value.replace("/", r"/\allowbreak{}")
    value = value.replace("-", r"-\allowbreak{}")
    return value


def main() -> None:
    if sha256(FREEZE) != EXPECTED_FREEZE:
        raise RuntimeError("Frozen v11 hash changed")
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(endpoint_rows())
    frame.to_csv(OUT / "endpoint_harmonization.csv", index=False)
    eppvr = inspect_eppvr()
    (OUT / "eppvr_metadata_trace.json").write_text(json.dumps(eppvr, indent=2, ensure_ascii=False), encoding="utf-8")
    paths = sorted({part.strip() for value in frame.provenance_paths for part in value.split(";")})
    structured = {"schema_version": "dcs-opct-v11-endpoint-harmonization-1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_sha256": sha256(FREEZE),
        "rows": frame.to_dict(orient="records"),
        "source_sha256": {path: sha256(ROOT / path) for path in paths if (ROOT / path).is_file()},
        "claim_boundary": "Heterogeneous endpoint contracts are recorded, not declared equivalent or pooled."}
    (OUT / "endpoint_harmonization.json").write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = (frame.groupby(["dataset", "ontology", "mapping_rule", "evidence_role", "comparability_boundary"],
                             sort=False).task.agg(lambda values: "/".join(values)).reset_index())
    lines = [r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{1.70cm}>{\raggedright\arraybackslash}p{3.10cm}>{\raggedright\arraybackslash}p{3.25cm}>{\raggedright\arraybackslash}p{5.00cm}@{}}",
             r"\caption{Dataset-specific endpoint harmonization. Rules and boundaries are fixed within each dataset; semantic differences are recorded without asserting construct equivalence. AMIGOS and Emognition are protocol-only because participant archives remain unavailable.}\label{tab:endpoint-harmonization}\\", r"\toprule",
             "Dataset & Endpoint and ontology & Fixed mapping & Comparability boundary \\\\", r"\midrule"]
    for item in compact.itertuples(index=False):
        values = [item.dataset, f"{item.task}; {item.ontology}", item.mapping_rule, item.comparability_boundary]
        lines.append(" & ".join(tex_escape(str(value)) for value in values) + " \\\\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (OUT / "table_endpoint_harmonization.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    md = """# Endpoint harmonization and EPPVR metadata trace

The study contains participant-rated dimensional endpoints, fixed stimulus/segment-category endpoints, and protocol-only reserved endpoints. These are not one interchangeable psychological outcome. DCS-OPCT transports configuration-level material identity-optimism probability, but all evidence remains conditional on each dataset's fixed task ontology.

EPPVR is locally verified as 30 participants x 14 trials x 10 synchronized channels x 7000 samples at 100 Hz, with FP1/FP2 in channels 0/1 and a 10-s per-trial baseline. Device identity, reference/ground, channels 5-9, the third label column, ethics approval, consent, and privacy authorization require author confirmation.
"""
    (OUT / "endpoint_harmonization_and_eppvr_trace.md").write_text(md, encoding="utf-8")
    tracked = [OUT / name for name in ("endpoint_harmonization.csv", "endpoint_harmonization.json",
        "eppvr_metadata_trace.json", "table_endpoint_harmonization.tex", "endpoint_harmonization_and_eppvr_trace.md")]
    manifest = {"status": "endpoint_harmonization_built_pending_independent_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_sha256": sha256(FREEZE),
        "row_count": len(frame), "dataset_count": frame.dataset.nunique(),
        "files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Endpoint harmonization built: {len(frame)} rows across {frame.dataset.nunique()} datasets")


if __name__ == "__main__":
    main()
