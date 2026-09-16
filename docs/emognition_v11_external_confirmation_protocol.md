# Emognition DCS-OPCT v11 External Replication Protocol

## Status and role

Emognition was reserved on 2026-09-09 before participant-value access as a
second independent wearable-EEG test of frozen DCS-OPCT v11. It is not a
replacement selected after seeing AMIGOS outcomes. Both datasets remain in the
evidence ledger, and every acquired one-shot result must be reported. A
replicated external-effectiveness claim requires both datasets to be acquired,
endpoint-eligible, non-identity releases, and complete passes; an acquisition
failure, identity fallback, or non-estimable endpoint is not a success.

## Cohort and endpoint

All 43 publicly described participants are attempted. The primary trials are
the ten shared STIMULUS-phase videos; baseline, washout, and questionnaire-phase
signals are excluded. For each task, a participant needs at least six usable
stimulus trials, both classes, the four registered channels, and at least three
valid windows per retained trial. At least 30 participants are required.

The tasks are experienced valence and experienced arousal from the published
nine-point SAM ratings. Ratings 6--9 are high, 1--4 are low, and midpoint 5 is
excluded. No sample-derived threshold or class-balancing relabeling is allowed.
One participant-video aggregate is one trial.

## Wearable EEG representation

The registered channels are TP9, AF7, AF8, and TP10 at 256 Hz. The bilateral
pairs are AF7--AF8 and TP9--TP10. Primary construction uses common-average
reference, four-second windows with two-second step, linear detrending, Welch
PSD, fixed theta/alpha/beta/gamma integration, log power, relative power,
log-difference, normalized asymmetry, and mean/population-SD/least-squares-slope
summaries. The representations are all, relative power, and normalized
asymmetry. Baseline normalization is not part of the one-shot primary analysis.

## Counterfactual and GPU execution

Five subject folds and five two-video stimulus folds are generated for each of
the five frozen seeds. Logistic regression and the 40-epoch CUDA MLP are used
at doses 0, 0.25, 0.5, 0.75, and 1 under the existing matched constraints and
0.02 material-event threshold. If construction is complete, the two tasks,
three representations, two models, five seeds, and four nonzero doses produce
240 configuration outcomes. Long-running stages display progress bars.

The pre-access endpoint-eligibility operating-characteristic analysis is
reported separately in `emognition_v11_endpoint_eligibility_planning.md`. It
quantifies hypothetical event-variation risk only and is not an empirical
prevalence estimate, power result, or acceptance gate.

## Frozen decision and interpretation

The unchanged five-predictor risk model and unchanged DCS-OPCT v11 gates are
applied. Audit assignment and the unlabeled action are locked before material
events are attached. Both halves require event and non-event observations.
Only a non-identity action with an audit Brier-gain lower bound above 0.001,
held-out Brier gain above 0.001, AUROC delta at least -0.02, rank at least
0.999999, and zero inversions supports dataset-specific prospective
effectiveness with bounded released-action non-harm. This cannot establish
universal safety.

## Authorized next action

The authorized next action is to send the signed official EULA from an academic
email associated with the Harvard Dataverse account to `emotions@pwr.edu.pl`.
After written authorization, the official `study_data.zip` should be placed at
`E:/AA发表论文的数据/dataset/Emognition`. No participant file may be opened until
the schema-only manifest, synthetic adapter tests, and implementation lock are
complete.

The pre-access schema decisions are fixed in
`emognition_v11_schema_adapter_preaccess_specification.md`. Raw Muse values
remain on their native numerical scale without an undocumented unit conversion,
and only complete four-second windows with the headband on and all four HSI
indicators in the published good-or-medium states are retained. An unknown JSON
nesting, timestamp alignment, unit convention, or quality code is a schema
failure rather than an invitation to tune the adapter after access.

The dataset-neutral table, dose, blind-lock, endpoint, and claim semantics are
implemented under the pre-access
`external_wearable_v11_standardized_analysis_contract.md`. The eventual
Emognition adapter may resolve only official-schema details; it may not replace
or relax the shared analysis core.
