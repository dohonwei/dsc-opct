# EKM-ED DCS-OPCT v11 External Confirmation Protocol

## Status and role

EKM-ED was reserved on 2026-09-16 before archive download or participant-value
access as an openly accessible, sparse wearable-EEG test of frozen DCS-OPCT
v11. Public Zenodo metadata, the archive name, byte size, checksum, license,
folder description, cohort size, stimuli, and public key-moment timestamps were
inspected before registration. No archive member, participant identifier, SAM
row, EEG sample, derived feature, or model outcome was accessed.

EKM-ED does not replace AMIGOS or Emognition after an observed outcome. Their
access failures remain in the evidence ledger. Every EKM-ED acquisition,
structural, endpoint, action, effectiveness, and non-harm result must also be
retained. EKM-ED can support only a dataset-specific prospective result; it
cannot by itself support replicated effectiveness or universal safety.

## Cohort and endpoint

All 47 publicly described participants are attempted. Primary trials are the
anger, sadness, happiness, and fear films; the baseline film is excluded. The
two tasks are participant-experienced valence and arousal from the official SAM
table. Integer ratings 6--9 are high, 1--4 are low, and midpoint 5 is excluded.
If the official questionnaire does not unambiguously provide this nine-point
participant-by-film structure, the affected task fails without substitute
labels.

For each task, a participant needs at least three retained film trials, both
classes, all four channels, and at least three valid four-second key-moment
windows in every retained trial. At least 30 participants are required. One
participant-film aggregate is one trial; samples and windows are never treated
as independent observations.

## Wearable EEG representation

Primary construction uses the official preprocessed clean Muse branch sampled
at 128 Hz. The registered channels are TP9, AF7, AF8, and TP10. Only the public
clip-specific key-moment timestamps are used. Complete four-second windows
centered on those timestamps undergo common-average referencing, linear
detrending, Welch PSD, fixed theta/alpha/beta/gamma integration, and the frozen
wearable summary construction. The three representations are all features,
relative power, and normalized asymmetry. The baseline film and full-clip
signals are not primary replacements if key-moment alignment fails.

## Counterfactual and GPU execution

Five subject folds and four physical-stimulus folds are generated for each of
the five frozen seeds. Logistic regression and the 40-epoch CUDA MLP are run at
doses 0, 0.25, 0.5, 0.75, and 1 under the existing matched constraints and the
0.02 material-event threshold. A complete design contains 240 nonzero-dose
configuration outcomes across two tasks, three representations, two models,
five seeds, and four nonzero doses. Long-running stages display progress bars.

## Frozen decision and interpretation

The unchanged five-predictor risk model and DCS-OPCT v11 gates are applied.
Audit assignment and the unlabeled action are locked before material events are
attached. Both audit and held-out partitions require event and non-event
observations. Only a non-identity action with an audit Brier-gain lower bound
above 0.001, held-out Brier gain above 0.001, AUROC delta at least -0.02, rank
at least 0.999999, and zero inversions supports dataset-specific prospective
effectiveness with bounded released-action non-harm. Identity is non-
intervention, not a safety or effectiveness success.

## Authorized next action

The authorized next action is to download the official Zenodo archive to
E:/AA发表论文的数据/dataset/EKM-ED without opening participant members, verify
the registered size and MD5, and record only its ZIP central directory. Adapter
implementation and locking must occur before any participant file is parsed.
Unknown questionnaire columns, channel names, timestamp units, directory
nesting, or key-moment alignment are schema failures rather than reasons to
tune the protocol after access.
