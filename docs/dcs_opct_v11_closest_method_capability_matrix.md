# DCS-OPCT v11 closest-method capability matrix

Date checked: 2026-09-10

## Scope

This is a focused novelty audit against the closest methodological families,
not a claim that every calibration, selective-prediction, conformal, or domain-
adaptation paper has been exhaustively reviewed. The comparison is based on the
methods' stated problem formulations and primary papers.

## Capability comparison

| Method family | Primary target | Counterfactual identity-utilization endpoint | Unlabeled target applicability gate | Limited-label release certificate | Order-preserving correction | Domain-level return-original fallback |
|---|---|---:|---:|---:|---:|---:|
| CPCS / calibrated prediction under covariate shift | Instance predictive probabilities under covariate shift | No | Uses unlabeled target data for shift correction, not a fail-closed applicability decision | No target-label release certificate | Not the central guarantee | No |
| TransCal | Instance confidence calibration for a domain-adapted classifier | No | Uses unlabeled target information, not an explicit abstain-before-certificate gate | No target-label release certificate | Temperature-based calibration can preserve prediction order | No |
| ATC | Dataset-level target accuracy estimation from unlabeled target predictions | No | No intervention-release gate | No | Not a probability correction | No |
| SelectiveNet / selective classification | Instance-level prediction versus rejection | No | Rejects instances by learned selection, not domains by transport applicability | No cross-domain calibration certificate | Not applicable | Rejects an instance rather than retaining a frozen domain-level probability map |
| Conformal risk control | Expected monotone loss or prediction-set risk using a calibration sample | No | Distribution-shift extensions require explicit assumptions; no identity-mechanism applicability witness | Yes, for its specified risk target | Not the central guarantee | No return-original audit-probability action |
| Better-practices domain-adaptation validation | Selection and validation of adaptation algorithms without target-test leakage | No | Evaluates validation criteria rather than releasing a probability correction | No | Not applicable | No |
| DCS-OPCT v11 | Configuration-level probability of material identity-induced evaluation optimism | Yes | Yes: displacement, order, and complementary-witness checks occur before outcome-based release | Yes: complete-cluster limited-label Brier/AUROC certificate | Yes: positive-slope map with verified rank preservation | Yes: identity is explicitly non-intervention |

## Defensible novelty statement

DCS-OPCT does not claim novelty for covariance alignment, quantile mapping,
logistic calibration, selective prediction, or risk certification individually.
Its new capability is the closed operational link between:

1. a counterfactual configuration-level endpoint that distinguishes identity
   opportunity, encoding, and classifier utilization;
2. cross-domain prediction of material evaluation optimism;
3. label-free screening of whether a probability correction is admissible;
4. a limited-label, complete-cluster release certificate; and
5. explicit return of the frozen probabilities when the action is unsupported.

The 2026-09-10 mechanism-falsification analysis adds evidence for the first
link: within-dataset encoding quartiles had a Q4-minus-Q1 event-rate contrast of
0.025 with a 95% complete-cluster bootstrap interval crossing zero, whereas
opportunity delta and metadata-prior opportunity had positive contrasts of
0.076 and 0.153 with intervals excluding zero. Additive and full mechanism
models improved leave-one-dataset-out AUROC over encoding-only in six of seven
domains; the exact dataset-level sign-randomization p-value was 0.03125 for
both. The interaction term was not superior to the additive model and must not
be presented as independently validated novelty.

## Claim boundary

The matrix supports a capability-difference argument and a high-level
engineering novelty claim. It does not establish prospective external
effectiveness, universal non-harm, conformal coverage, causal invariance, or
exhaustive priority over every related method.

## Primary sources

- Park et al. (2020), Calibrated Prediction with Covariate Shift via
  Unsupervised Domain Adaptation, AISTATS. https://proceedings.mlr.press/v108/park20b.html
- Wang et al. (2020), Transferable Calibration with Lower Bias and Variance in
  Domain Adaptation, NeurIPS. https://papers.nips.cc/paper/2020/hash/df12ecd077efc8c23881028604dbb8cc-Abstract.html
- Garg et al. (2022), Leveraging Unlabeled Data to Predict Out-of-Distribution
  Performance, ICLR. https://arxiv.org/abs/2201.04234
- Geifman and El-Yaniv (2019), SelectiveNet: A Deep Neural Network with an
  Integrated Reject Option, ICML. https://proceedings.mlr.press/v97/geifman19a.html
- Angelopoulos et al. (2022), Conformal Risk Control. https://arxiv.org/abs/2208.02814
- Ericsson et al. (2023), Better Practices for Domain Adaptation, AutoML.
  https://proceedings.mlr.press/v224/ericsson23a.html
- Ben-David et al. (2010), Impossibility Theorems for Domain Adaptation,
  AISTATS. https://proceedings.mlr.press/v9/david10a.html
