# DCS-OPCT v11 Manuscript Architecture and Argument Blueprint

Date: 2026-09-09
Target: Biomedical Signal Processing and Control
Evidence status: retrospective development plus post-access exploratory robustness; no successful prospectively reserved external effectiveness confirmation

Prospective status: AMIGOS and Emognition are reserved as a two-member external
wearable-EEG family, but neither authorized archive has been acquired and no
participant-level values or outcomes have been accessed. The family registry
forbids outcome-based dataset selection. These reservations are reproducibility
safeguards, not completed experiments or sources of empirical support.
The shared implementation now also forces the mechanism design and unlabeled
action/audit assignment to be hashed before emotion-classifier fitting creates
material-event outcomes; this is a prospective integrity control, not evidence
that either external endpoint will be eligible or successful.

## Recommended title

**From Identity Exposure to Selective Cross-Domain Calibration: A Risk-Aware Audit Framework for EEG Emotion Recognition**

Method-forward alternative: **DCS-OPCT: Distribution-Covered, Order-Preserving Calibration of Identity-Shortcut Audit Risk Across Affective-Computing Domains**.

The first title is recommended because it presents the complete scientific arc rather than only the final calibrator. It also avoids implying that the method adapts the emotion classifier itself.

## Central thesis

Identity overlap in repeated-measures affective data is not merely a split-design nuisance: controlled exposure can change apparent performance and model ranking, a mechanism-derived model can prioritize configurations at risk of material optimism, and a selective cross-domain calibrator can improve those risk probabilities when target support is adequate while abstaining when it is not.

## Problem-to-solution logic

1. EEG emotion models are often selected on repeated-measures datasets in which participant identity can occur in both training and test data.
2. A performance gap between split protocols does not identify why optimism occurs, because opportunity, feature encoding, and downstream use are confounded.
3. The counterfactual dose experiment varies identity-label opportunity while fixing test rows, training size, binary class counts, and identity coverage. It isolates dose-induced amplification of the exposure effect.
4. Material amplification can alter which model appears best and create deployment regret under joint-unseen evaluation. This converts a methodological concern into model-selection risk.
5. A frozen mechanism-derived model can rank risky configurations, but the EPPVR test shows that ranking may transfer while probabilities and operating thresholds do not.
6. DCS-OPCT addresses this failure mode. It transports frozen audit-risk probabilities, not EEG emotion predictions, and releases correction only after unlabeled support checks, independent-transform agreement, and a distribution-covered labeled audit.
7. When support is inadequate, identity is the scientifically correct action. Abstention is part of the method, not an effectiveness result.

## Research questions

- **RQ1, mechanism:** When identity-label opportunity is increased under matched training conditions, does the apparent benefit of identity exposure increase, and are opportunity, encoding, and utilization empirically separable?
- **RQ2, consequence:** Can identity exposure reverse model ranking and cause regret when the selected model is evaluated on jointly unseen identities?
- **RQ3, risk prediction:** Can a model restricted to mechanism-derived, dataset-agnostic predictors identify configurations with material dose-induced optimism across held-out public datasets and in EPPVR?
- **RQ4, selective calibration:** Can target-domain risk probabilities be corrected without changing ranking, and can unsupported corrections be rejected before release?
- **RQ5, boundary:** Does independent wearable physiology support effectiveness, or only demonstrate that fixed applicability gates can abstain under severe shift?
- **Prospective confirmation question (not yet answered):** On the independently
  collected 14-channel AMIGOS and four-channel Emognition wearable-EEG
  protocols, will the frozen framework encounter eligible mixed endpoints and,
  if so, release corrections that pass the unchanged certificate and held-out
  criteria independently?

## Contributions

1. A counterfactual identity-dose design that separates identity opportunity, identity encoding, and classifier utilization under fixed test observations and matched training constraints.
2. A model-selection consequence analysis linking identity exposure to material winner reversal, rank inversion, and joint-unseen deployment regret.
3. A frozen audit-risk model trained only on mechanism-derived predictors, with dataset, task, representation, and model names explicitly forbidden.
4. DCS-OPCT, combining a CORAL primary transform, a quantile-mapping witness, fixed displacement and rank gates, distribution-covered half-audit assignment, and stratified cluster certification.
5. A transparent validation chain retaining successful corrections, principled abstentions, predecessor failures, and the failed AVDOS and FACED registered confirmation branches.

The AMIGOS pre-access reservation is not listed as a scientific contribution:
until the one-shot execution is completed, it is an open confirmatory protocol
and contributes no result.

The same boundary applies to Emognition. Neither reservation, nor the external-
family registry that prevents best-dataset reporting, is an empirical
contribution before execution.

## Claim hierarchy

Primary claim: within the evaluated development support region, DCS-OPCT selectively improved calibration of a frozen identity-shortcut audit-risk model while preserving ranking, and returned identity in unsupported domains.

Supporting claims:

- Controlled identity opportunity can produce material dose-induced optimism.
- Exposure-biased evaluation can reverse model selection.
- The frozen risk score retained discrimination on EPPVR but failed calibration and threshold transport.
- Distribution-covered audits yielded positive held-out Brier gains in EPPVR, CASE, and CEAP.
- SEED-IV, DREAMER, and exploratory AVDOS-VR and FACED were rejected by fixed unlabeled gates.
- Exploratory FACED demonstrated strong participant encoding without a material event at the frozen endpoint, separating encoding from observed utilization while leaving calibration effectiveness non-estimable.

Forbidden extensions:

- DCS-OPCT improves EEG emotion-classification accuracy or is universal EEG domain adaptation.
- Identity fallback proves effectiveness.
- AVDOS-VR or FACED is an untouched or confirmatory success.
- The framework universally prevents negative transfer.
- The intervention identifies a physiological causal mechanism.

## IMRaD architecture

### 1. Introduction, approximately 1,000 words

Move from repeated-measures evaluation to a concrete cross-domain audit decision. Establish that split comparisons, identity probes, and metadata priors answer different questions. Introduce the second problem that discrimination can transfer without calibration, then state the three-stage mechanism-risk-safety loop and bounded claims.

### 2. Materials and methods, approximately 2,500 words

- **2.1 Study scope and evidence roles:** development, external test, exploratory robustness, and frozen components.
- **2.2 Stage I, counterfactual identity dose:** fixed test cell, matched control, exposure block, nominal dose, achieved opportunity, exposure effect, dose-induced amplification, and material event.
- **2.3 Model-ranking consequences:** winner reversal, material reversal, Kendall correlation, pairwise inversion fraction, and deployment regret.
- **2.4 Stage II, frozen audit-risk model:** five predictors, L2 logistic model, public-data training, forbidden identities, leave-dataset-out validation, and frozen threshold.
- **2.5 Transport endpoint:** Brier score and gain; explain why a positive-slope map preserves AUROC.
- **2.6 Stage III, DCS-OPCT:** CORAL and quantile transforms, positive-slope OPCT, component gates, witness agreement, and identity fallback.
- **2.7 Distribution-covered half-audit:** seven unlabeled geometry features, subset enumeration, coreset objective, label-free lock, and complete-cluster bootstrap certificate.
- **2.8 Datasets and computation:** evidence roles, five seeds, model and representation panels, 40-epoch CUDA MLP, 5,000 bootstraps, GPU, and deterministic manifests.

### 3. Results, approximately 2,000 words

- **3.1 Mechanism:** identity dose reveals heterogeneous and sometimes non-monotonic utilization.
- **3.2 Consequence:** exposure changes model selection and creates joint-unseen regret.
- **3.3 Risk score:** public leave-dataset-out transfer succeeds, while EPPVR discrimination transfers but calibration and threshold do not.
- **3.4 DCS-OPCT development:** EPPVR, CASE, and CEAP adapt; SEED-IV and DREAMER abstain.
- **3.5 Wearable boundary:** exploratory AVDOS-VR shows PPG identity decodability, non-monotonic utilization, excessive probability shifts, and identity fallback.
- **3.6 Encoding--utilization boundary:** exploratory FACED shows high identity decodability, zero material events at the frozen endpoint, excessive probability shifts, and a non-estimable calibration endpoint.

AMIGOS must not appear as a Results subsection before execution. If protocol
transparency is useful at submission time while access remains pending, mention
the reservation only in the reproducibility or future-validation paragraph and
state explicitly that it supplied no data to method development or claims.
Emognition follows the same rule. If both are eventually executed, each result
must be shown independently before any cross-dataset synthesis.

### 4. Discussion, approximately 1,500 words

Explain the closed loop, why this is not generic domain adaptation, why order preservation matters, and why abstention is valid non-intervention evidence but not effectiveness. State the retrospective status, small source-domain count, constructed event threshold, dataset-specific preprocessing, failed AVDOS and FACED registered confirmations, event-free FACED endpoint, and lack of a successful prospectively reserved compatible external dataset.

### 5. Conclusion, approximately 250 words

Restate the bounded contribution and separate selective development evidence from external confirmation.

## Formula dossier

For configuration $c$ and dose $d\in\{0,0.25,0.5,0.75,1\}$,

$$E_{c,d}=\operatorname{BAcc}_{c,d}^{\mathrm{exposed}}-\operatorname{BAcc}_{c}^{\mathrm{matched\ control}},$$

$$A_{c,d}=E_{c,d}-E_{c,0},\qquad Y_{c,d}=\mathbb{1}(A_{c,d}\ge 0.02).$$

The frozen risk score is

$$p_{c,d}=\sigma\!\left(\beta_0+\boldsymbol{\beta}^{\top}\mathbf{x}_{c,d}\right),$$

where $\mathbf{x}$ contains opportunity change, metadata-prior opportunity, encoding margin, opportunity-by-encoding interaction, and model capacity.

Positive-slope OPCT is

$$T_{a,b}(p)=\sigma\!\left[a\,\operatorname{logit}(p)+b\right],\qquad a>0.$$

Because $a>0$, the transform preserves strict orderings and ties.

For primary and witness displacements $\Delta^{(P)}$ and $\Delta^{(W)}$,

$$G_{\mathrm{dir}}=|I|^{-1}\sum_{i\in I}\mathbb{1}(\Delta_i^{(P)}\Delta_i^{(W)}\ge0),$$

$$G_{\cos}=\frac{\langle\Delta^{(P)},\Delta^{(W)}\rangle}{\|\Delta^{(P)}\|_2\|\Delta^{(W)}\|_2},$$

$$G_{\mathrm{dis}}=n^{-1}\sum_i\frac{|\Delta_i^{(P)}-\Delta_i^{(W)}|}{|\Delta_i^{(P)}|+|\Delta_i^{(W)}|+\epsilon}.$$

Release requires $G_{\mathrm{dir}}\ge0.90$, $G_{\cos}\ge0.90$, and $G_{\mathrm{dis}}\le0.35$, plus component shift, rank, and inversion gates.

The distribution-covered subset minimizes

$$J(S)=\|\boldsymbol\mu_S-\boldsymbol\mu_G\|_2^2+0.05\|\boldsymbol\sigma_S-\boldsymbol\sigma_G\|_2^2.$$

Brier score and gain are

$$\operatorname{BS}(p)=n^{-1}\sum_i(p_i-y_i)^2,\qquad G_{\mathrm{BS}}=\operatorname{BS}(p^{\mathrm{id}})-\operatorname{BS}(p^{\mathrm{DCS}}).$$

The audit certificate requires the one-sided familywise lower confidence bound of $G_{\mathrm{BS}}$ to exceed 0.001. AUROC noninferiority has margin $-0.02$ and is analytic only after rank $\ge0.999999$ and zero inversions are verified.

## Figure plan

1. Three-stage mechanism-risk-safety framework. A new schematic is still needed.
2. Counterfactual dose and ranking consequences. A new combined publication figure is still needed.
3. Six-dataset action matrix: fig_v11_six_dataset_action_matrix.pdf.
4. Audit and held-out gain: fig_v11_audit_and_heldout_gain.pdf.
5. Support-gate geometry: fig_v11_support_gate_geometry.pdf.
6. AVDOS exploratory boundary: fig_avdos_exploratory_identity_and_dose.pdf.
7. FACED exploratory external boundary: fig_faced_post_access_external_boundary.pdf.
8. Failure and protocol timeline: supplementary figure only.

## Reviewer-facing stress test

| Likely objection | Required answer |
|---|---|
| This is only another calibration method. | Calibration is the final stage of a mechanism-to-decision framework. The target event is experimentally defined, linked to model-selection harm, and calibrated selectively. |
| The method uses target labels. | The action and audit assignment are locked without labels. A bounded labeled half-audit certifies release; the other half remains held out. This is limited-label validation, not unsupervised adaptation. |
| Identity fallback trivially cannot hurt. | Correct. Fallback supports non-intervention safety only and is never counted as effectiveness. |
| Development data were repeatedly inspected. | All predecessor failures are retained, v11 is labeled retrospective development evidence, and no external-confirmation claim is made. |
| AVDOS is not confirmatory. | Correct. The confirmatory branch stopped before model fitting; the completed analysis is post-access exploratory PPG robustness only. |
| FACED is not confirmatory. | Correct. The registered 32-channel branch was structurally ineligible. The amended 30-channel analysis is post-access exploratory and its event-free endpoint cannot estimate calibration effectiveness. |
| AMIGOS is listed, so is it already external validation? | No. AMIGOS was reserved before participant-value access, and only schema-independent software tests have been completed. The archive has not been acquired, no outcomes have been inspected, and it contributes no empirical evidence at present. |
| Why add Emognition after AMIGOS? | It was reserved before any member participant values were accessed to test a substantially sparser four-channel wearable device. A family registry forbids choosing the better outcome: every acquired one-shot result is retained, and replicated effectiveness requires both datasets to pass independently. |
| Why not use direct target Platt scaling? | DCS-OPCT is designed for a fixed audit budget and requires unlabeled support agreement plus held-out verification. A direct-calibration comparator should be added when a compatible external dataset is obtained. |

## Submission readiness

The current evidence supports a BSPC manuscript centered on selective audit-risk calibration, principled abstention, and the experimentally observed distinction between identity encoding and utilization. It does not support prospectively confirmed safe and effective cross-domain transfer. The highest-value additional experiment remains one compatible, prospectively reserved external dataset with adequate endpoint variation on which the frozen method either earns a positive certificate and held-out gain or abstains under the locked gate.

AMIGOS and Emognition are the currently reserved external family. Their pre-
access registrations, schema-only inspection boundaries, feature-equivalence
tests, GPU implementation checks, pre-outcome action-lock ordering, and
no-selection registry reduce analytical flexibility. The current validation
counts are 13/13 for AMIGOS and 13/13 for Emognition. Both members now also have
50,000-repetition hypothetical endpoint-eligibility analyses under the same
60-cluster design and a tested three-stage lock state machine. These are software and protocol
checks only: they do not alter submission readiness or the empirical claim until
the authorized one-shot runs have been completed and frozen, including all null,
failed, non-estimable, or abstained outcomes.
