# DCS-OPCT

Reproducibility materials for the manuscript **From Identity Exposure to Selective Cross-Domain Calibration of Audit-Risk Probabilities in EEG Emotion Recognition**.

## Scope

This repository contains source code, frozen protocols, aggregate evidence, validation reports, manuscript sources, figures, and tables for DCS-OPCT. It does not redistribute restricted EEG source datasets or reversible participant-level records.

This GitHub repository is a public, author-identifiable reproducibility mirror; it is not an anonymized review repository.

DCS-OPCT calibrates a configuration-level audit-risk probability. It does not alter trial-level EEG emotion predictions or retrain the underlying emotion classifier.

## Validated submission archive

The locally validated release archive is `dcs_opct_v11_submission_artifacts_crossfit_v22.zip`.

SHA-256:

```text
442BFB517EA8DD88E0693080780946EF9A321AFC66CE8F756DE4243A6E19AC66
```

The archive-level verification covered 538/538 entries with no CRC failure or manifest mismatch. The embedded submission-package validation passed 26/26 checks, the T1–T11 traceability validation passed 16/16 checks, and the EKM-ED confirmatory structural-failure validation passed 19/19 checks.

## Claim boundary

The evidence supports retrospective selective configuration-level audit-risk calibration within evaluated development support and verifies fail-closed execution. The prospectively reserved EKM-ED attempt was structurally ineligible before model fitting. These results do not establish prospective external effectiveness, universal transport safety, future-domain non-harm, or direct improvement of EEG emotion predictions.

## Environment

- Python 3.11
- CUDA-capable PyTorch runtime for GPU experiments
- Dependencies in `requirements.txt` and `requirements-experiment-lock.txt`
- LaTeX with `latexmk`, `pdflatex`, and BibTeX for manuscript compilation

See `docs/dcs_opct_v11_anonymous_repository_readme.md` for the full validation commands and data-governance boundaries. The filename is retained to match the immutable v22 archive manifest; its "anonymous" label describes the prepared review package rather than this public GitHub mirror.
