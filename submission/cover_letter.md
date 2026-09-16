# Cover Letter

September 16, 2026

Editor-in-Chief  
Biomedical Signal Processing and Control

Dear Editor,

Please consider our manuscript, "From Identity Exposure to Selective Cross-Domain Calibration of Audit-Risk Probabilities in EEG Emotion Recognition," for publication as a research article in *Biomedical Signal Processing and Control*.

Repeated-measures EEG emotion-recognition studies can report optimistic performance when participant identity overlaps training and test sets. Existing split comparisons and identity probes do not isolate whether an emotion classifier actually uses identity-related opportunity, and they do not indicate when the resulting evaluation bias is large enough to change model selection. Our study addresses this engineering problem through a three-stage, auditable workflow.

First, a counterfactual identity-dose experiment varies identity--label opportunity while fixing tested observations, training size, class counts, and identity coverage. Second, a mechanism-derived model estimates configuration-level risk of material optimism and links this risk to winner reversal and deployment regret. Third, DCS-OPCT conditionally recalibrates that audit-risk probability using bounded, order-preserving transport, an unlabeled applicability screen, and a limited-label release certificate. The method changes neither EEG emotion predictions nor classifier parameters.

The main methodological contribution is the closed release-or-return-original decision path, rather than a claim that CORAL, quantile mapping, or logistic calibration is individually new. Matched-budget experiments compare this path with direct, split-certified, and cross-fit-certified calibrators, as well as configuration-level CPCS-style and ATC-style neighbors. The study also reports endpoint-measurement sensitivity, cluster-aware inference, leave-one-component diagnostics, and an explicit independence ladder.

We have deliberately bounded the claims. Positive calibration evidence is retrospective and strongest for EPPVR after target-covariate holdout and raw test-participant separation. When all EPPVR participants and training rows were divided into two independently rebuilt populations, both candidate shifts exceeded the frozen applicability limit and the method returned the original probability. This verifies fail-closed execution but is not presented as successful fully independent calibration or universal safety. A separately reserved and implementation-locked EKM-ED attempt failed the frozen participant-level endpoint-eligibility contract before model fitting, providing a prospective endpoint-transport failure rather than an effectiveness result. No completed external dataset produced a prospective non-identity release; this limitation is stated in the Abstract, Discussion, Conclusion, and Supplementary Material.

The manuscript is relevant to the journal because it studies biomedical signal-processing evaluation under identity dependence, probability reliability under dataset shift, limited-label calibration, and reproducible release control. The package includes GPU-enabled code, progress reporting, frozen hashes, failure records, aggregate results, and independent validation reports. Restricted source data are excluded and remain governed by their original access conditions.

This manuscript is original, is not under consideration elsewhere, and has been approved by all authors. The authors declare no competing interests. The EPPVR protocol received ethics approval from the Ethics Committee of the University of Electronic Science and Technology of China (approval no. 106142023122227999), and written informed consent and authorization for secondary analysis were obtained.

Thank you for considering our work.

Sincerely,

Zhiqi Huang  
Corresponding author  
School of Automation Engineering  
University of Electronic Science and Technology of China  
Chengdu 611731, China  
202121060701@std.uestc.edu.cn
