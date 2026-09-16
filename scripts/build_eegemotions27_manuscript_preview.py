from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/elsarticle"
EEG_ROOT = ROOT / "outputs/eegemotions27_v11_posthoc_boundary_analysis_v2"
OUT = ROOT / "outputs/eegemotions27_v11_manuscript_integration_preview"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {name} anchor, found {count}")
    return text.replace(old, new, 1)


def fit_table_to_textwidth(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    begin = "\\begin{tabular}"
    end = "\\end{tabular}"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError(f"Expected exactly one tabular in {path}")
    text = text.replace(begin, "\\resizebox{\\textwidth}{!}{%\n" + begin, 1)
    text = text.replace(end, end + "%\n}", 1)
    path.write_text(text, encoding="utf-8")


def prepare_tree() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    tables = OUT / "tables"
    if figures.exists():
        shutil.rmtree(figures)
    if tables.exists():
        shutil.rmtree(tables)
    shutil.copytree(SOURCE / "figures", figures)
    shutil.copytree(SOURCE / "tables", tables)

    for name in ["elsarticle.cls", "elsarticle-num.bst", "dcs_opct_v11_bspc_refs.bib"]:
        shutil.copy2(SOURCE / name, OUT / name)

    figure_source = EEG_ROOT / "figures"
    for name in [
        "fig_eegemotions27_applicability_boundary.pdf",
        "fig_eegemotions27_dual_generalization.pdf",
        "fig_eegemotions27_identity_vs_affect.pdf",
        "fig_eegemotions27_dose_heterogeneity.pdf",
        "fig_eegemotions27_gate_flow.pdf",
    ]:
        shutil.copy2(figure_source / name, figures / name)

    table_source = EEG_ROOT / "latex_tables_v4"
    for name in [
        "table_eegemotions27_gate.tex",
        "table_eegemotions27_applicability.tex",
        "table_eegemotions27_dual_generalization.tex",
        "table_eegemotions27_encoding_boundary.tex",
    ]:
        shutil.copy2(table_source / name, tables / name)
    for name in [
        "table_eegemotions27_applicability.tex",
        "table_eegemotions27_dual_generalization.tex",
    ]:
        fit_table_to_textwidth(tables / name)


def build_manuscript() -> Path:
    source_path = SOURCE / "dcs_opct_v11_bspc_manuscript.tex"
    text = source_path.read_text(encoding="utf-8")

    abstract_anchor = (
        "A separate post-access FACED stress test also failed the applicability gate and "
        "contained no material event at the frozen endpoint, so calibration effectiveness "
        "was not estimable."
    )
    abstract_addition = (
        abstract_anchor
        + " In a separate pre-signal EEGEmotions-27 robustness test, both unlabeled "
        "projections exceeded the frozen probability-displacement limit and the procedure "
        "returned identity before target labels entered action selection."
    )
    text = replace_once(text, abstract_anchor, abstract_addition, "abstract")
    text = replace_once(
        text,
        "Across 100 matched-budget repetitions, no DCS-OPCT release produced material "
        "negative transfer.",
        "Across 100 matched-budget retrospective repetitions, no DCS-OPCT release "
        "produced material negative transfer.",
        "abstract retrospective boundary",
    )

    methods_anchor = (
        "Together, the AVDOS and FACED chronology prevents either post-access result from "
        "being presented as external confirmation."
    )
    methods_addition = methods_anchor + r"""

EEGEmotions-27 was evaluated under a separate evidence role. Repository-structure metadata, including participant-by-emotion availability encoded in filenames, had been inspected before registration, but no EEG sample, participant demographic row, derived feature, prediction, endpoint event, or model outcome had been accessed. The analysis was then locked before signal and outcome access at repository commit \texttt{0dfec14d177ed73c37a95f06d942e9b58e2bbf77}. It is therefore termed a separate pre-signal external robustness test rather than a pristine prospective confirmation, and it is not a member of the AMIGOS/Emognition prospective family.

The registered EEGEmotions-27 task used fixed stimulus-category polarity: emotion-video IDs 4, 20, 22, and 25 were positive, whereas IDs 5, 6, 13, 17, 18, and 24 were negative. These labels describe the assigned stimulus category and are not participant-experienced valence or arousal. One feature row represented one participant--physical-video pair after within-trial aggregation. The fixed grid used the 14-channel Emotiv X montage, three representations, two classifiers, five split seeds, five subject folds, four physical-stimulus folds, and five identity doses. Both participant and physical-video identities were held out in the dual-unseen endpoint. All GPU MLP fits used the frozen 40-epoch CUDA implementation, and the action and distribution-covered audit assignment were hashed before category-polarity outcomes entered the release decision."""
    text = replace_once(text, methods_anchor, methods_addition, "methods evidence role")

    controls_anchor = (
        "Independent validators reproduced the nine development checks, seventeen "
        "leave-one-component checks, twelve AVDOS exploratory checks, eighteen FACED "
        "exploratory checks, and 78 submission-artifact checks."
    )
    controls_addition = (
        "Independent validators reproduced the nine development checks, seventeen "
        "leave-one-component checks, twelve AVDOS exploratory checks, eighteen FACED "
        "exploratory checks, twelve EEGEmotions-27 one-shot checks, six post-hoc artifact "
        "checks, and 78 submission-artifact checks."
    )
    text = replace_once(text, controls_anchor, controls_addition, "validation inventory")

    contribution_anchor = (
        "and returned identity in four unsupported development or exploratory domains. "
        "It does not directly improve EEG emotion-classifier accuracy"
    )
    contribution_addition = (
        "and returned identity in four unsupported development or exploratory domains. "
        "A separate pre-signal external robustness test also returned identity before "
        "target labels entered action selection. It does not directly improve EEG "
        "emotion-classifier accuracy"
    )
    text = replace_once(
        text,
        contribution_anchor,
        contribution_addition,
        "contribution evidence-role count",
    )

    results_block = r"""
\subsection{EEGEmotions-27 exposed a pre-signal external applicability boundary}

The registered EEGEmotions-27 analysis retained 87 participants and 864 participant--physical-video trials after the fixed structural rules. Three representations, two classifiers, five split seeds, five subject folds, four stimulus folds, and five nominal doses produced 150 summary rows and 3,600 CUDA fitting units. At zero dose, all six dual-unseen category-polarity balanced-accuracy means were below 0.50, ranging from 0.400 to 0.473. In contrast, subject-identity encoding margins ranged from 0.722 to 0.942. Thus, substantial participant information coexisted with weak transfer of the registered stimulus-category target when both identities were unseen. These descriptive results concern category-derived polarity and cannot be interpreted as recognition of experienced valence or arousal.

The release decision failed before category-polarity outcomes entered action selection. Absolute mean probability displacements were 0.1073 for CORAL plus OPCT and 0.1281 for quantile mapping plus OPCT, exceeding the frozen 0.10 limit by 0.0073 and 0.0281, respectively. Both maps retained rank 1.0 with zero inversions, showing that excessive probability displacement, rather than order failure, caused inapplicability. The witness stage was skipped, the candidate and selected method were identity, and no labeled release certificate was attempted. The complete one-shot reconstruction passed all 12 independent checks.

\begin{figure*}[t]
\centering
\includegraphics[width=0.78\textwidth]{figures/fig_eegemotions27_applicability_boundary.pdf}
\caption{Separate pre-signal EEGEmotions-27 applicability boundary. Both unlabeled projections crossed the frozen 0.10 probability-displacement maximum, so no correction was eligible for release.}
\label{fig:eegemotions-applicability}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_eegemotions27_dual_generalization.pdf}
\caption{EEGEmotions-27 category-polarity balanced accuracy when both participants and physical stimuli were held out. Points are the five frozen split seeds; error bars show one seed-level standard deviation for description only. Category polarity is stimulus-derived and is not participant-experienced valence or arousal.}
\label{fig:eegemotions-dual-generalization}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_eegemotions27_gate_flow.pdf}
\caption{Frozen EEGEmotions-27 action path. Excessive unlabeled probability displacement triggered identity fallback before target labels entered action selection. Identity is non-intervention: no action was released, and neither effectiveness nor safety success is claimed.}
\label{fig:eegemotions-gate}
\end{figure*}

\input{tables/table_eegemotions27_gate.tex}

Post-hoc analyses were performed only after this immutable decision. Across 30 representation--model--axis configurations, none showed a strong globally monotonic dose response; model rankings reversed in 6 of 60 comparisons and representation rankings in 12 of 40. The two non-released candidate maps also had negative held-out Brier gains. These observations help explain the failed boundary but cannot retrospectively justify the 0.10 limit, convert identity fallback into effectiveness, or promote EEGEmotions-27 into the registered prospective family.

"""
    text = replace_once(
        text,
        "\\section{Discussion}\n",
        results_block + "\\section{Discussion}\n",
        "results insertion",
    )

    identity_anchor = (
        "Treating abstention or non-estimability as effectiveness would erase the distinction "
        "on which the method is built."
    )
    identity_addition = identity_anchor + (
        " EEGEmotions-27 sharpened this boundary because the action failed on unlabeled "
        "probability displacement before category-polarity labels entered selection. Its "
        "identity result therefore demonstrates protocol adherence under an independently "
        "collected wearable-EEG geometry, not external calibration benefit."
    )
    text = replace_once(text, identity_anchor, identity_addition, "discussion boundary")

    limitation_anchor = (
        "Both completed analyses are therefore exploratory. The decisive remaining experiment "
        "is a compatible external dataset reserved before participant-level values are accessed "
        "and containing sufficient endpoint variation."
    )
    limitation_addition = (
        "Both completed analyses are therefore exploratory. EEGEmotions-27 was locked before "
        "signal and outcome access after repository metadata disclosure, but both unlabeled "
        "projections were inapplicable; it supplies a separately tested failure boundary rather "
        "than effective external release. The decisive remaining experiment is a compatible "
        "external dataset reserved before participant-level values are accessed and containing "
        "sufficient endpoint variation."
    )
    text = replace_once(text, limitation_anchor, limitation_addition, "limitations")

    conclusion_anchor = (
        "and rejected the post-access FACED transport after both components exceeded the "
        "displacement budget."
    )
    conclusion_addition = conclusion_anchor + (
        " The separate pre-signal EEGEmotions-27 test also returned identity after CORAL and "
        "quantile probability shifts exceeded the same frozen limit."
    )
    text = replace_once(text, conclusion_anchor, conclusion_addition, "conclusion")

    text = replace_once(
        text,
        "AVDOS-VR, and FACED remain subject to their original access conditions.",
        "AVDOS-VR, FACED, and EEGEmotions-27 remain subject to their original access conditions.",
        "data availability",
    )
    text = replace_once(
        text,
        "successful external effectiveness on AVDOS-VR or FACED,",
        "successful external effectiveness on AVDOS-VR, FACED, or EEGEmotions-27,",
        "conclusion external-effectiveness boundary",
    )

    output = OUT / "manuscript_preview.tex"
    output.write_text(text, encoding="utf-8")
    return output


def build_supplement() -> Path:
    source_path = SOURCE / "dcs_opct_v11_bspc_supplementary.tex"
    text = source_path.read_text(encoding="utf-8")

    scope_anchor = (
        "The completed AVDOS-VR PPG and FACED common-montage EEG analyses are post-access "
        "exploratory and do not restore either unsuccessful registered confirmation branch."
    )
    scope_addition = scope_anchor + (
        " EEGEmotions-27 is a separate pre-signal external robustness test conducted after "
        "repository-structure metadata disclosure; it is outside the AMIGOS/Emognition "
        "prospective family and returned identity before target outcomes entered action selection."
    )
    text = replace_once(text, scope_anchor, scope_addition, "supplement scope")

    section = r"""
\section{EEGEmotions-27 pre-signal robustness boundary}

EEGEmotions-27 was pinned at repository commit \texttt{0dfec14d177e}; the complete commit identifier is retained in the registration and implementation lock. Registration followed inspection of the public README, channel metadata, and repository filenames, but preceded access to EEG sample values, participant demographics, derived features, predictions, or outcomes. The evidence role is therefore separate pre-signal robustness, not pristine prospective confirmation. The registered task maps fixed physical emotion-video categories to positive or negative polarity; it does not measure participant-experienced valence or arousal.

The structural analysis retained 87 participants and 864 participant--stimulus trials. The frozen grid produced 150 summary rows and 3,600 CUDA fitting units. The action and distribution-covered 15/15 audit--held-out cluster assignment were hashed before target outcomes entered the decision. CORAL plus OPCT shifted the mean probability by 0.1073 and quantile mapping plus OPCT by 0.1281; both exceeded the 0.10 applicability maximum. The witness and labeled certificate were therefore skipped and identity was retained.

\input{tables/table_eegemotions27_applicability.tex}
\input{tables/table_eegemotions27_dual_generalization.tex}
\input{tables/table_eegemotions27_encoding_boundary.tex}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=0.78\textwidth]{figures/fig_eegemotions27_identity_vs_affect.pdf}
\caption{Post-hoc EEGEmotions-27 encoding--utilization boundary. Subject-identity encoding margins were high, while dual-unseen category-polarity balanced accuracy remained below chance on average for every representation--classifier combination. The figure is descriptive and cannot alter the frozen action.}
\label{fig:supp-eegemotions-identity}
\end{figure*}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_eegemotions27_dose_heterogeneity.pdf}
\caption{Post-hoc EEGEmotions-27 dose heterogeneity. Curves show mean exposure effects across five frozen seeds and shading shows one seed-level standard deviation. No evaluated configuration exhibited a strong globally monotonic dose pattern; these diagnostics explain heterogeneity but do not rescue the failed external gate.}
\label{fig:supp-eegemotions-dose}
\end{figure*}

The nonreleased held-out counterfactuals were unfavorable: CORAL plus OPCT had Brier gain $-0.0349$ and AUROC change $-0.0509$, while quantile mapping plus OPCT had Brier gain $-0.0569$ and AUROC change $-0.1200$. These values were unavailable to the pre-outcome applicability decision and are reported only to characterize the rejected actions. They cannot be used to retune the threshold or claim that abstention proved future-domain safety.

"""
    text = replace_once(
        text,
        "\\section{Reproducibility and claim boundary}\n",
        section + "\\section{Reproducibility and claim boundary}\n",
        "supplement results insertion",
    )

    validation_anchor = (
        "18 of 18 FACED exploratory checks, and 78 of 78 submission-artifact checks."
    )
    validation_addition = (
        "18 of 18 FACED exploratory checks, 12 of 12 EEGEmotions-27 one-shot checks, "
        "6 of 6 EEGEmotions-27 post-hoc artifact checks, and 78 of 78 "
        "submission-artifact checks."
    )
    text = replace_once(text, validation_anchor, validation_addition, "supplement validation")

    bullet_anchor = (
        "\\item FACED is post-access exploratory evidence only; the registered 32-channel "
        "branch was structurally ineligible."
    )
    bullet_addition = bullet_anchor + (
        "\n\\item EEGEmotions-27 is a separate pre-signal robustness test outside the "
        "AMIGOS/Emognition prospective family; category polarity is stimulus-derived and the "
        "identity result is not effectiveness or safety success."
    )
    text = replace_once(text, bullet_anchor, bullet_addition, "supplement claim bullet")

    itemize_anchor = "\\begin{itemize}\n\\item The transported endpoint"
    itemize_replacement = (
        "\\begin{itemize}\n"
        "\\setlength{\\itemsep}{1pt}\n"
        "\\setlength{\\parskip}{0pt}\n"
        "\\setlength{\\parsep}{0pt}\n"
        "\\item The transported endpoint"
    )
    text = replace_once(
        text,
        itemize_anchor,
        itemize_replacement,
        "supplement compact claim-boundary list",
    )

    output = OUT / "supplementary_preview.tex"
    output.write_text(text, encoding="utf-8")
    return output


def write_manifest(manuscript: Path, supplement: Path) -> None:
    tracked = [manuscript, supplement]
    tracked.extend(sorted((OUT / "figures").glob("fig_eegemotions27_*.pdf")))
    tracked.extend(sorted((OUT / "tables").glob("table_eegemotions27_*.tex")))
    manifest = {
        "status": "staged_preview_not_merged_into_authoritative_manuscript",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manuscript": "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
        "source_supplement": "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex",
        "source_files_unchanged": True,
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in tracked
        },
        "claim_boundary": (
            "This preview integrates a failed external robustness boundary. It does not "
            "establish external effectiveness, non-harm, safety, or replication."
        ),
    }
    (OUT / "preview_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    prepare_tree()
    manuscript = build_manuscript()
    supplement = build_supplement()
    write_manifest(manuscript, supplement)
    print(f"Staged manuscript: {manuscript}")
    print(f"Staged supplement: {supplement}")


if __name__ == "__main__":
    main()
