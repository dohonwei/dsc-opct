# AMIGOS DCS-OPCT v11 External Confirmation Protocol

## Status and evidence boundary

AMIGOS is prospectively reserved on 2026-09-09 for one frozen DCS-OPCT v11
evaluation. No AMIGOS participant value, trial label, EEG sample, derived
feature, or model outcome had been accessed when the reservation was written.
The official archive is not currently present locally. Receipt of the archive
does not authorize analysis until a schema-only acquisition manifest and a
hashed implementation lock have both been written.

This test may fail at acquisition, eligibility, construction, endpoint
variation, applicability, certification, effectiveness, or released-action
non-harm. Every failure remains part of the evidence. Identity fallback is
non-intervention, not successful calibration or a safety success.

## Cohort, stimuli, and tasks

Only the individual-viewing branch containing the 16 shared short videos is
eligible. All released participants are attempted. A participant is retained
only if all 16 physical-video intervals and all 14 registered EEG channels can
be verified and each trial yields at least three valid windows. At least 30
complete participants are required before model fitting.

The two tasks use the fixed published stimulus quadrants: expected valence
(high versus low) and expected arousal (high versus low), with eight videos in
each class for each task. Participant self-reports are not used to define the
confirmatory labels. The labels therefore concern intended stimulus quadrant,
not necessarily experienced emotion. One participant-video aggregate is one
trial; windows are never independent observations.

## Signal and feature construction

The required montage is AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4,
F8, and AF4. Signal units and sampling rate must be verified from the official
schema before values are converted. The same fixed feature definitions used in
the FACED implementation are applied: common-average reference, four-second
windows with two-second step, linear detrending, Welch PSD, theta/alpha/beta/
gamma integration, relative power, log power, and normalized asymmetry for the
seven registered bilateral pairs. Window sequences are summarized by mean,
population standard deviation, and least-squares slope. The three frozen
representations are all features, relative power only, and normalized
asymmetry only.

## Counterfactual identity dose and GPU execution

The primary identity axis is participant identity. Four physical-stimulus
folds and five subject folds are generated for each of the five frozen split
seeds. The same 256-candidate, class-matched, subject-complete exposure-block
construction selects nominal doses 0, 0.25, 0.5, 0.75, and 1. Test rows,
training size, binary class counts, and exposure of held-out identities remain
fixed. The models are logistic regression and the 40-epoch CUDA MLP. CUDA is a
hard requirement for the registered neural branch, and every long stage must
show a `tqdm` progress bar.

Two tasks, three representations, two models, five seeds, and four nonzero
doses yield 240 configuration-level material-event outcomes if construction is
complete. The frozen event remains dose-induced balanced-accuracy
amplification at least 0.02. No threshold or endpoint may be changed if the
event count is inconvenient.

## One-shot transport and acceptance

The unchanged frozen risk model produces the five mechanism predictors and the
configuration-level material-event probability. The unchanged DCS-OPCT v11
component, witness, displacement, agreement, monotonicity, distribution-
covered audit, bootstrap, and acceptance thresholds are then applied. The
audit assignment and unlabeled action are written and hashed before material-
event outcomes are attached.

Both the audit and held-out partitions must contain an event and a non-event;
otherwise calibration effectiveness is non-estimable. A positive external
result additionally requires a non-identity action, audit Brier-gain lower
bound above 0.001, held-out Brier gain above 0.001, held-out AUROC delta at
least -0.02, probability rank at least 0.999999, and zero order inversions.
Only this complete conjunction can support prospective external effectiveness
with bounded released-action non-harm. It cannot establish universal safety.

## Authorized next action

The next authorized action is acquisition of the official AMIGOS archive under
its EULA and placement at `E:/AA发表论文的数据/dataset/AMIGOS`. Credentials,
account cookies, and private download links must not be stored in this
repository or passed to the analysis scripts.

The dataset-neutral table, dose, blind-lock, endpoint, and claim semantics are
implemented under the pre-access
`external_wearable_v11_standardized_analysis_contract.md`. The eventual AMIGOS
adapter may resolve only official-schema details; it may not replace or relax
the shared analysis core.
