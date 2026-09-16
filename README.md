# DCS-OPCT

Reproducibility materials for the manuscript **From Identity Exposure to Selective Cross-Domain Calibration of Audit-Risk Probabilities in EEG Emotion Recognition**.

## Scope

This repository contains source code, frozen protocols, aggregate evidence, validation reports, manuscript sources, figures, and tables for DCS-OPCT. It does not redistribute restricted EEG source datasets or reversible participant-level records.

This GitHub repository is a public, author-identifiable reproducibility mirror; it is not an anonymized review repository.

DCS-OPCT calibrates a configuration-level audit-risk probability. It does not alter trial-level EEG emotion predictions or retrain the underlying emotion classifier.

## Validated submission archive

The current validated release archive is `dcs_opct_v11_submission_artifacts_crossfit_v25.zip`. Earlier archives remain available as immutable predecessors.

The authoritative archive digest is stored in the adjacent
`dcs_opct_v11_submission_artifacts_crossfit_v25.zip.sha256` sidecar.

The v25 archive-level verification covered 547/547 entries with no CRC failure or manifest mismatch. The embedded submission-package validation passed 28/28 checks, the T1–T11 traceability validation passed 16/16 checks with no author-dependent submission blocker, and the EKM-ED confirmatory structural-failure validation passed 19/19 checks.

## Claim boundary

The evidence supports retrospective selective configuration-level audit-risk calibration within evaluated development support and verifies fail-closed execution. The prospectively reserved EKM-ED attempt was structurally ineligible before model fitting. These results do not establish prospective external effectiveness, universal transport safety, future-domain non-harm, or direct improvement of EEG emotion predictions.

## Environment

- Python 3.11
- CUDA-capable PyTorch runtime for GPU experiments
- Dependencies in `requirements.txt` and `requirements-experiment-lock.txt`
- LaTeX with `latexmk`, `pdflatex`, and BibTeX for manuscript compilation

See `docs/dcs_opct_v11_public_repository_readme.md` for the current validation commands and data-governance boundaries. The historical anonymized-package filename is retained only for compatibility with the immutable v22 manifest.
