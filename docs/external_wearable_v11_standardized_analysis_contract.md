# External Wearable EEG v11 Standardized Analysis Contract

## Status and scope

This contract was written on 2026-09-09 before participant-value access to
either prospectively registered wearable-EEG family member. It defines the
dataset-neutral interface between a schema-specific adapter, the
counterfactual identity-dose experiment, and frozen DCS-OPCT v11. It does not
change the frozen model, material threshold, transport action, audit rule, or
acceptance gate, and it supplies no external empirical evidence.

## Standard trial table

One row represents one retained participant by physical-stimulus trial after
all windows have been aggregated. Window-level observations are never passed
to a classifier as independent samples. The required columns are:

| Field | Contract |
|---|---|
| `subject_id` | Non-empty stable participant identifier; used only for subject-wise splitting and identity-opportunity construction. |
| `trial_id` | Stable integer identifier for the same physical stimulus across participants; ordinal row position is not an acceptable substitute. |
| task label | Registered binary label for the current task. It must be formed by the dataset-specific rule before the shared core is called. |
| feature columns | Finite trial-level values grouped into exactly the registered `all`, `relative_power`, and `normalized_asymmetry` representations. |

Each `(subject_id, trial_id)` pair must be unique. The number of distinct
subjects and stimuli must be at least the registered subject- and stimulus-fold
counts. Missing trials are permitted only when the dataset-specific protocol
permits them; duplicate aggregates are not permitted.

## Counterfactual identity-dose invariants

For every split seed and crossed subject-by-stimulus test cell, the jointly
unseen quadrant is the control training set. A class-matched part of that set
is removed and replaced by a subject-complete block drawn from rows containing
the held-out subjects but different physical stimuli. Candidate blocks are
searched without changing test rows, total training size, binary class counts,
or coverage of held-out subject identities.

Let \(Z\) denote subject identity, \(Y\) the registered binary task label, and
\(B_d\) the selected exposure block at nominal dose \(d\). The achieved
opportunity is based on normalized mutual information,

\[
O(B_d)=\frac{I(Z;Y)}{H(Y)},
\]

and the normalized achieved dose is

\[
\tilde d=\frac{O(B_d)-O_{\min}}{O_{\max}-O_{\min}},
\]

with zero assigned when the candidate range is numerically degenerate. The
registered grid is \(d\in\{0,0.25,0.5,0.75,1\}\). The zero-dose block is an
empirical candidate-range anchor, not a no-exposure condition; all dose blocks
retain complete held-out-subject coverage.

The material configuration outcome at nonzero dose is

\[
M_d=\mathbb{1}\left[(\Delta BA_d-\Delta BA_0)\ge 0.02\right],
\]

where \(\Delta BA_d\) is the balanced-accuracy exposure effect relative to the
matched jointly unseen training set. The 0.02 threshold is immutable.

## Unlabeled mechanism table

Before any `exposure_effect`, dose-induced amplification, or material-event
value is loaded, the shared application core constructs one row per nonzero
configuration from five frozen predictors:

1. opportunity increase relative to the dose-zero anchor;
2. metadata-prior opportunity;
3. cross-stimulus subject-identity encoding margin;
4. opportunity-by-encoding interaction;
5. fixed ordinal model-capacity level.

Dataset, task, representation, model, cohort size, and row count are not risk
predictors. They remain grouping keys only. The frozen public-data scaler and
logistic model produce configuration-level material-event probabilities.

## Blinded action and assignment lock

The application order is mandatory:

1. Construct and hash the crossed dose plans, split audit, identity-encoding
   table, and pre-outcome mechanism summary before emotion-classifier fitting.
2. Read the pre-outcome mechanism summary, which contains no classifier
   probability, exposure effect, dose-induced amplification, or material event.
3. Compute identity, CORAL, and quantile-mapping risk probabilities.
4. Fit both positive-slope OPCT projections on CUDA for the frozen 500 epochs.
5. Apply displacement, rank, inversion, and independent-witness gates.
6. Construct the distribution-covered audit assignment from unlabeled
   probability geometry.
7. Write the component diagnostics, geometry, assignment, and blinded
   probabilities, then hash all four artifacts and the action-lock JSON.
8. Only after the lock hash exists, train the registered emotion classifiers,
   write `exposure_effect`, and derive material events.

The post-outcome summary must reproduce every pre-outcome mechanism quantity
after alignment on the complete configuration-dose key. The outcome-attachment
function verifies the action-lock hash both before and after reading outcomes.
Any mechanism mismatch or lock mismatch terminates the analysis.

## Endpoint and claim semantics

Both audit and held-out partitions must contain at least one event and one
non-event. If either partition has a single class, effectiveness is
non-estimable and the reported action is identity. This is not a safety
success.

If the unlabeled candidate is inapplicable or the audit certificate fails,
identity is a non-intervention. It is not evidence of calibration
effectiveness, released-action non-harm, or safety. Dataset-specific external
support requires the complete conjunction of a non-identity release, audit
certification, held-out Brier gain above 0.001, AUROC non-inferiority, frozen
rank preservation, and zero inversions. The resulting statement is limited to
prospective external effectiveness with bounded released-action non-harm; it
does not establish universal safety.

## Executable implementation

The shared dose implementation is
`scripts/external_wearable_dose_core.py`, and its CUDA invariant test is
`scripts/test_external_wearable_dose_core.py`. The shared blind-lock and
certification implementation is
`scripts/external_wearable_dcs_opct_core.py`, and its CUDA end-to-end test is
`scripts/test_external_wearable_dcs_opct_core.py`. Dataset-specific adapters
may supply trial tables and labels but may not alter these shared scientific
rules after participant values are accessed.

Before dose construction, every dataset-specific adapter output is checked by
`scripts/external_wearable_trial_contract.py`. The gate rejects duplicate
participant--stimulus aggregates, non-integer physical-stimulus identifiers,
missing or non-binary labels, nonfinite features, absent registered
representations, insufficient participants or stimuli, and violations of the
dataset-specific per-participant label rules. AMIGOS additionally requires the
same complete 16-video set and the same fixed 8/8 stimulus labeling for every
participant; Emognition requires at least six retained non-midpoint trials and
both classes for every retained participant-task bundle. Synthetic failure
injection is implemented independently in
`scripts/test_amigos_v11_adapter_contract.py` and
`scripts/test_emognition_v11_adapter_contract.py`.

After the schema-specific adapter and one-shot runner have been implemented,
`scripts/external_wearable_implementation_lock.py` binds their hashes to the
reservation, schema-only manifest, unchanged v11 freeze, shared dependencies,
passing synthetic tests, CUDA runtime, and an explicit no-participant-value-
access attestation. Dataset entry points are
`scripts/lock_amigos_v11_external_confirmation.py` and
`scripts/lock_emognition_v11_external_confirmation.py`. They refuse missing
schema-bound code, failed tests, absent CUDA, missing attestation, and overwrite
of an existing lock. The lock itself supplies no external empirical evidence.

The schema-specific runner must pass its validated task bundles to
`scripts/external_wearable_one_shot_core.py`. This shared entry point verifies
the implementation lock before each task, runs the dataset-specific adapter
contract, constructs the registered dose plans, and includes the per-task
adapter audits in the hashed pre-outcome bundle. Its synthetic integration test
is `scripts/test_external_wearable_one_shot_core.py`. Direct calls that omit
this adapter-audit chain are not authorized for the prospective datasets.

The complete synthetic execution test
`scripts/test_external_wearable_one_shot_end_to_end.py` uses the five frozen
split seeds, the frozen public risk model and risk table, CUDA OPCT, 240 real
logistic fits, registered material-event construction, and final endpoint
release semantics. It verifies the end-to-end order from schema-bound
implementation lock through pre-outcome design, blinded action, model outcomes,
and the final gate. Synthetic passage is implementation evidence only.

The resumable execution state machine is
`scripts/external_wearable_v11_pipeline.py`. It serializes and hashes the real
prepared dose-design objects together with the pre-outcome mechanism and split
audit tables, verifies that all serialized and tabular views are identical,
then creates a separately hashed action bundle. Its registered model-outcome
runner verifies both locks and every locked action artifact before every task,
performs the prepared classifier fits with the existing progress bars, and
verifies the action bundle again after all tasks finish. Finalization rejects
any change to either lock, any locked action artifact, or any mechanism quantity
in the post-outcome summary.
Its failure-injection test is
`scripts/test_external_wearable_v11_pipeline.py`.
