# DCS-OPCT v11 Historical Anonymized Review Package

This filename is retained for compatibility with the immutable v22 manifest.
The current public, author-identifiable repository instructions and publication
receipt are in `docs/dcs_opct_v11_public_repository_readme.md` and
`docs/dcs_opct_v11_public_repository_receipt_20260916.json`.

This package accompanies the BSPC manuscript "From Identity Exposure to Selective Cross-Domain Calibration of Audit-Risk Probabilities in EEG Emotion Recognition."

## Scope

The package reproduces the recorded configuration-level analyses, statistical summaries, figures, and validation checks. It does not redistribute restricted EEG source data. Dataset access remains governed by the original providers, institutional approval, consent, and data-use agreements.

The transported quantity is the probability that an evaluation configuration exhibits material identity-induced optimism. DCS-OPCT does not alter trial-level EEG emotion predictions or retrain an emotion classifier.

## Claim boundary

The evidence supports retrospective selective calibration within evaluated development support and fail-closed return-original behavior when frozen applicability conditions fail. It does not establish universal safe transfer, positive fully training-data-independent calibration, prospective external effectiveness, or future-domain non-harm.

## Environment

- Python 3.11
- CUDA-capable PyTorch runtime for GPU experiments
- Dependencies in `requirements.txt` and `requirements-experiment-lock.txt`
- LaTeX with `latexmk`, `pdflatex`, and BibTeX for manuscript compilation

## Primary validation commands

Run from the repository root:

```powershell
& 'D:\conda_envs\py311\python.exe' scripts\validate_dcs_opct_v11_eppvr_raw_partition_first.py
& 'D:\conda_envs\py311\python.exe' scripts\validate_dcs_opct_v11_reviewer_closure.py
& 'D:\conda_envs\py311\python.exe' scripts\validate_dcs_opct_v11_reproducibility_inventory.py
& 'D:\conda_envs\py311\python.exe' scripts\validate_dcs_opct_v11_reviewer_task_traceability.py
& 'D:\conda_envs\py311\python.exe' scripts\validate_ekmed_v11_confirmatory_failure.py
& 'D:\conda_envs\py311\python.exe' scripts\validate_dcs_opct_v22_submission_artifacts.py
```

The recorded v22 expected results are 21/21, 14/14, 23/23, 16/16, 19/19, and all v22 package checks passed, respectively. Use the public-repository README for the current v23 finalization command.

## Manuscript compilation

```powershell
Set-Location docs\elsarticle
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build dcs_opct_v11_bspc_manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build dcs_opct_v11_bspc_supplementary.tex
```

## Restricted-data reconstruction

Raw-data reconstruction requires authorized local copies of the named datasets. The scripts fail closed when required inputs, schema contracts, frozen hashes, CUDA, or implementation locks are absent. Do not replace missing datasets with unofficial mirrors or alter the frozen action thresholds after inspecting outcomes.

## Package integrity

`submission_artifact_manifest.json` records SHA-256 hashes for every packaged file. `independent_validation_report.json` verifies the hashes, manuscript freshness, clean LaTeX logs, required analysis reports, the T1--T11 traceability matrix, claim guardrails, and the exact author-only blockers that remain before journal submission.
