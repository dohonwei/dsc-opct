# DCS-OPCT v11 Public Reproducibility Repository

Repository: https://github.com/dohonwei/dsc-opct

This public, author-identifiable repository accompanies the BSPC manuscript
"From Identity Exposure to Selective Cross-Domain Calibration of Audit-Risk
Probabilities in EEG Emotion Recognition." It is not an anonymous review
repository.

## Scope and claim boundary

The repository reproduces the recorded configuration-level analyses,
statistical summaries, figures, package checks, and fail-closed evidence. It
does not redistribute restricted EEG source data or the designated
participant-level governance-sensitive outputs.

DCS-OPCT transports and selectively calibrates the probability that an
evaluation configuration exhibits material identity-induced optimism. It does
not alter trial-level EEG emotion predictions or retrain an emotion classifier.
The evidence supports retrospective selective calibration within evaluated
development support and verifies return-original behavior when frozen gates
fail. It does not establish universal safe transfer, positive fully
training-data-independent calibration, prospective external effectiveness, or
future-domain non-harm.

## Public release verification

The publication receipt is stored in
`docs/dcs_opct_v11_public_repository_receipt_20260916.json`. The immutable v22
archive has SHA-256
`442BFB517EA8DD88E0693080780946EF9A321AFC66CE8F756DE4243A6E19AC66`.
Its archive-level verification covered 538/538 entries with no CRC failure or
manifest mismatch.

## Primary validation commands

Run from the repository root with Python 3.11:

```powershell
& 'D:\conda_envs\py311\python.exe' scripts\validate_dcs_opct_v11_eppvr_raw_partition_first.py
& 'D:\conda_envs\py311\python.exe' scripts\validate_dcs_opct_v11_reviewer_closure.py
& 'D:\conda_envs\py311\python.exe' scripts\validate_dcs_opct_v11_reproducibility_inventory.py
& 'D:\conda_envs\py311\python.exe' scripts\validate_dcs_opct_v11_reviewer_task_traceability.py
& 'D:\conda_envs\py311\python.exe' scripts\validate_ekmed_v11_confirmatory_failure.py
& 'D:\conda_envs\py311\python.exe' scripts\validate_dcs_opct_v23_submission_artifacts.py
```

GPU experiment branches require a CUDA-capable PyTorch runtime. Long-running
training and bootstrap stages expose progress bars. Raw-data reconstruction
requires authorized local copies of each source dataset and fails closed when
required inputs, schemas, frozen hashes, CUDA, or implementation locks are
absent.
