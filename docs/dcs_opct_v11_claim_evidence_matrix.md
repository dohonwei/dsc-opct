# DCS-OPCT v11 Claim-Evidence Matrix

Date: 2026-09-16

## Defensible central claim

DCS-OPCT v11 is a selective cross-domain calibration procedure for a frozen
identity-shortcut audit-risk model. It releases a monotone CORAL-based
probability correction only when an unlabeled quantile witness agrees with the
primary transform, the probability displacement remains within a fixed budget,
and a distribution-covered labeled audit certifies Brier improvement without
AUROC loss. Within the evaluated development support region, it improved
held-out Brier score on EPPVR, CASE, and CEAP. It returned identity on SEED-IV,
DREAMER, post-access exploratory AVDOS-VR, post-access exploratory FACED, and
the separate pre-signal EEGEmotions-27 external robustness test.
The prospectively locked EKM-ED attempt stopped before model fitting because
the frozen key-moment endpoint yielded no eligible participant in either task.

This is not evidence that DCS-OPCT directly improves an EEG emotion classifier.
The transported endpoint is the probability of a configuration-level material
identity-optimism event produced by the frozen audit-risk model.

## Claim matrix

| Claim | Direct evidence | Admissible wording | Boundary or forbidden extension | Status |
|---|---|---|---|---|
| Controlled identity exposure can change apparent emotion-classifier performance. | Counterfactual dose experiments on public datasets and EPPVR; fixed test cells and matched training interventions. | "Identity opportunity was experimentally varied while the tested observations and matched training constraints were held fixed." | Do not call the intervention a neural or physiological causal mechanism. | Supported |
| Identity-related evaluation optimism can reverse model selection. | Model-ranking consequence outputs quantify material winner reversal and deployment regret. | "Exposure-biased evaluation can select a different model from joint-unseen evaluation." | Do not imply every dataset or model family reverses. | Supported with heterogeneity |
| A mechanism-derived score can rank audit risk across domains. | Frozen public-data score achieved EPPVR AUROC 0.6658 and AP/prevalence lift 2.192; absolute calibration failed. | "The score retained audit-priority discrimination under predictor shift." | Do not claim transported probabilities or a universal threshold. | Supported, bounded |
| DCS-OPCT v11 can improve audit-risk probability calibration within support. | Audit Brier-gain LCBs: EPPVR 0.009553, CASE 0.011205, CEAP 0.001936; held-out gains: 0.026309, 0.048475, 0.008456; AUROC delta exactly 0 through monotonicity. | "Certified calibration gains were reproduced in the held-out halves of three development datasets." | These are retrospectively assembled development datasets, not untouched external confirmations. | Supported as development evidence |
| The method can abstain before unsupported transport. | SEED-IV and DREAMER were returned to identity by fixed unlabeled gates; AVDOS-VR, FACED, and EEGEmotions-27 exceeded the 0.10 probability-displacement budget and also returned identity. | "The applicability layer rejected unsupported corrections and preserved the identity prediction." | Identity fallback gives zero change by construction; it is non-intervention, not evidence of effectiveness or universal safety. EEGEmotions-27 post-hoc harmful-candidate diagnostics cannot retrospectively validate the threshold. | Supported as fail-closed behavior |
| The method has externally confirmed effectiveness. | No compatible one-shot external test produced an effective release. The registered AVDOS-VR branch stopped before training; the registered FACED 32-channel branch was structurally ineligible; the separate pre-signal EEGEmotions-27 robustness test completed but abstained before certification; and prospectively locked EKM-ED failed structural endpoint eligibility before model fitting. | No affirmative wording is currently admissible. | Never call AVDOS-VR, FACED, EEGEmotions-27, or EKM-ED an external effectiveness success. | Not supported |
| AVDOS-VR supplies useful mechanism evidence. | Post-access exploratory PPG analysis: 37 participants, 222 trials, four representations, five seeds, 3600 GPU fits; identity encoding is high for raw PPG but dose response is non-monotonic. | "Exploratory AVDOS-VR results show that identity decodability need not imply monotonic utilization." | Do not generalize to EEG, valence, clinical risk, or prospective deployment. | Exploratory only |
| FACED supplies a mechanism boundary. | Post-access 30-channel EEG analysis: 123 participants, 2952 trials, 4500 GPU fits; identity-probe balanced accuracy 0.885--0.992 versus chance 0.008; amplification -0.0166 to 0.0139; 0/120 material events. | "Exploratory FACED results show that strong participant encoding need not imply material utilization at the frozen endpoint." | Zero events make calibration effectiveness non-estimable. Do not call the result safe, effective, confirmatory, or evidence that identity never matters in FACED. | Exploratory only |
| EEGEmotions-27 supplies a pre-signal external robustness boundary. | The frozen 14-channel analysis retained 87 participants and 864 participant--stimulus trials. All six dual-unseen category-polarity balanced-accuracy means were below 0.50, while identity-encoding margins ranged from 0.7215 to 0.9416. CORAL and quantile probability shifts were 0.1073 and 0.1281, both above the frozen 0.10 maximum, so the action reverted to identity before outcome labels entered the decision. Independent validation passed 12/12 checks. | "In a separate pre-signal robustness test, the frozen applicability gate rejected both global corrections under excessive target-domain probability displacement." | Category polarity is stimulus-derived, not experienced valence or arousal. The test is not pristine pre-metadata confirmation, is not part of the AMIGOS/Emognition family, and did not establish effectiveness or safety success. Post-hoc dose, ranking, and non-released candidate results are explanatory only. | Externally tested boundary; no effective release |
| EKM-ED supplies a prospective endpoint-transport boundary. | The archive checksum, central directory, two header lines, adapter, synthetic tests, GPU runtime, and unchanged v11 gates were locked before participant-value access. The one-shot preparation attempted all 47 questionnaire participants, retained 86/188 three-window trials, and produced zero eligible participants for valence or arousal under the frozen three-trial, both-class rule. No model fit, action, material event, or effectiveness outcome was created. | "In a prospectively reserved and implementation-locked wearable-EEG test, the frozen key-moment endpoint was structurally non-estimable before model fitting." | Do not reinterpret endpoint failure as algorithmic ineffectiveness, effective abstention, non-harm, safety, or evidence favoring a post-access alignment change. Any later EKM-ED variant is exploratory and cannot replace failure record 001. | Prospective structural failure; no effectiveness result |
| AMIGOS provides prospective external confirmation. | A pre-access history audit, one-shot reservation, implementation protocol, schema inspector, frozen-equivalent feature core, adapter-output contract with synthetic failure injection, tested schema-bound implementation-lock generator, adapter-to-pre-outcome integration test, full frozen-action CUDA one-shot test, CUDA MLP test, shared two-stage dose/action-lock core, resumable lock state machine, and a 13/13 stack validation were completed before participant-value access. The official archive has not been acquired and no participant values or outcomes have been inspected. | "AMIGOS has been prospectively reserved for a future one-shot external confirmation under an unchanged v11 gate." | Do not report AMIGOS as an experiment, dataset result, external validation, successful confirmation, or evidence for applicability, effectiveness, non-harm, or safety until the authorized one-shot analysis is completed. | Prospectively reserved; no result |
| Emognition provides prospective external replication. | A pre-access history audit, one-shot reservation, archived official EULA and metadata receipt, four-channel feature core, ZIP schema-only inspector, fail-closed documented-schema adapter, field- and archive-level synthetic failure injection, staged GPU runner, tested schema-bound implementation-lock generator, adapter-to-pre-outcome integration test, full frozen-action CUDA one-shot test, CUDA MLP test, shared two-stage dose/action-lock core, resumable lock state machine, 50,000-repetition endpoint-eligibility planning, cross-dataset family registry, and a 15/15 stack validation were completed before participant-value access. The EULA-controlled archive has not been acquired. | "Emognition has been prospectively reserved as a second independent wearable-EEG test under the unchanged v11 gate." | Do not call it a result or external validation. The eligibility simulation uses hypothetical prevalence and is not power or empirical evidence. Do not select between AMIGOS and Emognition based on outcomes; replicated effectiveness requires both to pass independently. | Prospectively reserved; no result |
| The framework prevents negative transfer universally. | No material negative transfer occurred among released v11 actions; v10 and earlier failures are preserved. | "No released v11 action caused material negative transfer in the evaluated datasets." | Do not state a universal guarantee; identity fallback trivially has zero change and an untested domain may violate the gate assumptions. | Supported only in evaluated cases |

## Three-layer paper logic

1. **Mechanism layer:** counterfactual dose experiments establish that identity
   opportunity, identity encoding, and classifier utilization are distinct.
2. **Risk layer:** a frozen mechanism-derived model predicts which experimental
   configurations are likely to show material evaluation optimism; ranking can
   transfer even when absolute probabilities do not.
3. **Safety layer:** DCS-OPCT v11 diagnoses target support, applies a monotone
   probability correction only under witness agreement, certifies benefit using
   a distribution-covered audit, and otherwise abstains.

The scientific contribution is the closed loop across these layers. Presenting
only the v11 calibration numbers would reduce the work to another adapter
comparison; presenting only the dose experiment would omit the deployable audit
decision.

## Required manuscript language

- Use "audit-risk probability transport" or "audit-risk calibration," not
  generic "EEG domain adaptation."
- Use "retrospective development evidence" for EPPVR, CASE, CEAP, SEED-IV, and
  DREAMER in the v11 selection experiment.
- Use "post-access exploratory robustness analysis" for AVDOS-VR.
- Use "post-access exploratory common-montage stress test" for FACED, and state
  that the registered 32-channel branch failed structural eligibility.
- Use "separate pre-signal external robustness test" for EEGEmotions-27. State
  that it was locked after filename-level repository metadata disclosure, is
  outside the AMIGOS/Emognition prospective family, and returned identity after
  both unlabeled projections exceeded the frozen displacement budget.
- Describe EEGEmotions-27 labels as fixed category-derived polarity, never as
  participant-experienced valence or arousal. Treat its post-hoc dose,
  ranking-reversal, and non-released candidate analyses as boundary evidence
  that cannot alter the primary gate.
- Describe EKM-ED as a prospectively reserved, checksum-verified,
  implementation-locked structural endpoint attempt. Report 86/188 retained
  three-window trials and zero eligible participants for both tasks. State that
  execution stopped before model fitting and that this is non-estimability, not
  effective abstention, ineffectiveness, non-harm, or safety.
- Preserve `outputs/ekmed_v11_external_confirmation/failure_001_structural_ineligibility.json`
  as the only confirmatory EKM-ED result. Any alternate alignment, key-moment,
  completeness, label, or channel analysis is post-access exploratory and
  cannot overwrite it.
- State that positive-slope OPCT preserves ordering, so AUROC delta is exactly
  zero only after the rank and inversion gates pass.
- State that the 5000-repetition certificate resamples complete four-dose
  split-seed clusters within task-by-representation-by-model strata.
- Keep the CEAP v6, v7, v8, v9, v10, AVDOS execution failures, and both FACED
  structural eligibility failures visible in the supplement or reproducibility
  appendix.
- Cite `docs/avdos_v11_effective_protocol_status.md` whenever the AVDOS protocol
  chain is summarized; the base exploratory protocol's threshold-5 paragraph is
  superseded and is not the executed label definition.
- Treat AMIGOS as a protocol-status item only until authorized data are acquired
  and the locked one-shot analysis is completed. Its reservation and pre-access
  software validation belong in the reproducibility record, not in the Results
  section or the dataset count for completed experiments.
- Do not let the existence of a prepared AMIGOS implementation strengthen the
  empirical claim. Only the future frozen execution can change the external-
  confirmation status, and an ineligible endpoint or identity fallback will not
  count as effectiveness or safety success.
- Apply the prospective external-family registry to AMIGOS and Emognition. All
  acquired one-shot outcomes must be reported, and no pooled, best-dataset, or
  at-least-one-success endpoint is permitted. A replicated external claim
  requires both members to pass independently.
- Preserve the two-stage execution order: hash the mechanism design and lock the
  unlabeled action/audit assignment before fitting the registered emotion
  classifiers that generate exposure effects. Passing the synthetic CUDA tests
  demonstrates implementation readiness only, not external validity.

## Remaining decisive gap

A successful, prospectively reserved compatible external dataset with adequate
endpoint variation and an effective released action is still needed for the
strongest target-journal claim. The
current evidence supports a paper about selective calibration, principled
abstention, a pre-signal external failure boundary, a prospective structural
endpoint failure, and an exploratory
encoding--utilization boundary, but not a paper
claiming independently confirmed universal effective-and-safe transfer. AMIGOS
and Emognition are prospectively reserved members of a locked external family,
but at the present pre-access stage they contribute procedural credibility only
and no empirical evidence. EKM-ED contributes a completed negative prospective
test of endpoint transportability, not a positive calibration result.
