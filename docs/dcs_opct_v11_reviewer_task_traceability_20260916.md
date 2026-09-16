# DCS-OPCT v11 Reviewer Task Traceability

Date: 2026-09-16

Authoritative source: `tmp/nature_review_studio/review_dcs_opct_v11_bspc_20260914.json`.
This file distinguishes implemented closure, bounded partial resolution, author-dependent submission
items, and the remaining external scientific blocker. A fail-closed return-original action is never
counted as positive calibration effectiveness.

| ID | Source concern | Closure status | Resolution and remaining boundary |
|---|---|---|---|
| T1 | Held-out 为 transductive 且信息访问边界不清 | `RESOLVED_WITH_BOUNDED_CLAIM` | Added an inductive target-transform sensitivity with seed-only assignment and audit-side transform fitting. EPPVR and CASE retained positive held-out gains; CEAP returned original. Boundary: This removes held-out-covariate access but not all underlying training-data dependence; the stronger raw-partition-first EPPVR analysis is tracked under T2. |
| T2 | Audit 与 held-out 可能复用底层 trial | `RESOLVED_WITH_FAIL_CLOSED_RESULT` | Added raw-group endpoint reconstruction and a 15/15 EPPVR raw-partition-first rebuild. The latter used disjoint participants, raw rows, training rows, identity probes, and fitted emotion models, requiring 9,600 emotion-model fits and 160 identity-probe fits. Boundary: The fully disjoint analysis returned original at the applicability gate. It verifies fail-closed execution, not positive training-data-independent calibration. |
| T3 | 最近邻跨域校准与性能估计基线缺失 | `RESOLVED_WITH_FORMAL_NONCOMPARABILITY_BOUNDARY` | Implemented matched-resource CPCS-style calibration and native ATC-style target-accuracy estimation. TransCal was not forced into an invalid comparison because no target-adapted task classifier logits exist in the frozen configuration-risk pipeline. Boundary: The result is a concrete matched-contract comparison, not superiority over every CPCS or TransCal implementation. |
| T4 | 0.02 endpoint 未处理测量误差 | `RESOLVED_AS_MEASUREMENT_ERROR_SENSITIVITY` | Propagated endpoint measurement error with paired crossed bootstrap resampling and soft event probabilities, then repeated source-risk and target-action sensitivity analyses. Boundary: The sensitivity covers test-sample uncertainty, not repeated model fitting or all shared training-data uncertainty; the frozen binary endpoint was not re-optimized. |
| T5 | n_domain=2 且七域 gate 失败 | `PARTIALLY_RESOLVED_PROSPECTIVE_ATTEMPT_STRUCTURALLY_INELIGIBLE` | Narrowed every central claim to retrospective selective calibration within evaluated development support and reported the failed seven-domain gate. Completed a checksum-verified, schema-bound, implementation-locked EKM-ED prospective attempt without post-access retuning. The frozen key-moment endpoint yielded zero eligible participants in both tasks and stopped before model fitting. AMIGOS and Emognition remain reserved, and no compatible dataset has produced a positive non-identity external release. Boundary: Scientific blocker for any prospective external-effectiveness or universal-safety claim. It does not prevent submission of the explicitly bounded retrospective paper. |
| T6 | Figure 1 未显示 source/target/audit/held-out 信息流 | `RESOLVED` | Revised the framework figure and caption to distinguish source, target, audit, held-out, transductive, inductive, raw-group, and raw-partition-first information boundaries. Boundary: None beyond the claim limits already shown in the figure caption. |
| T7 | Table 1 二元能力判断过于自定义 | `RESOLVED` | Reframed Table 1 as a neutral resource and decision-level matrix reporting each method family's prediction object, target information, assumptions, and present-study implementation status. Boundary: The table is positioning evidence, not an empirical superiority table. |
| T8 | 无匿名复现仓库或 DOI | `RESOLVED_PUBLIC_REPOSITORY_VERIFIED` | Published the checksum-verified v22 reproducibility package, source code, aggregate outputs, and validation reports in a public author-identifiable GitHub repository. Independently verified public visibility, the remote commit, the SHA-256 sidecar, and the uploaded archive hash. Boundary: The GitHub repository is author-identifiable and is therefore suitable only when BSPC permits non-anonymous repository disclosure or after identity masking is no longer required. |
| T9 | Funding 声明缺失 | `RESOLVED_FROM_OFFICIAL_PROJECT_RECORDS` | Official project plan and approval records identify National Natural Science Foundation of China project No. 62172081; the same EPPVR study also acknowledges that project. A bounded Funding section was added without importing unrelated grants from the prior paper. Boundary: Only grant No. 62172081 is included; two Guangxi grants from the prior paper were excluded because their contribution to the present study was not established. |
| T10 | 局部结果措辞和摘要仍可压缩 | `RESOLVED` | Standardized outcome-held-out and transductive/inductive terminology, compressed comparator detail in the abstract, moved repeated diagnostics to the supplement, and aligned manuscript, highlights, and cover letter claim boundaries. Boundary: Further compression is editorial rather than a missing analysis. |
| T11 | Channels 5–9 缺精确索引 | `RESOLVED_UNAVAILABLE_TRANSPARENTLY_DISCLOSED` | Reported the verified collective composition of channels 5--9 and explicitly declined to guess their within-block index order. These channels are not used in the present analysis. Boundary: The exact within-block order is unavailable in the inspected acquisition and analysis records. This is explicitly disclosed and does not affect the executed FP1/FP2 analysis. |

## Evidence map

### T1: RESOLVED_WITH_BOUNDED_CLAIM

**Reviewer strategy:** 新增 cross-fitted/inductive target-transform sensitivity，并重画数据流

**Evidence files:**

- `outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/analysis_report.json`
- `outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/independent_validation_report.json`
- `outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/inductive_vs_transductive.csv`

**Manuscript anchors:**

- `We added a stricter configuration-level inductive sensitivity.`
- `The covariate-unseen sensitivity separated results that required transductive target-batch access`

### T2: RESOLVED_WITH_FAIL_CLOSED_RESULT

**Reviewer strategy:** 按共享 participant/stimulus/source-row family 重组或重采样

**Evidence files:**

- `outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914/analysis_report.json`
- `outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914/independent_validation_report.json`
- `outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/analysis_manifest.json`
- `outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/independent_validation_report.json`
- `docs/elsarticle/figures/fig_v11_independence_ladder.pdf`

**Manuscript anchors:**

- `We therefore added a second, nested raw-group sensitivity.`
- `Finally, we performed a raw-partition-first EPPVR sensitivity.`
- `No participant, global raw row, training row, or fitted target-domain emotion model crossed the population boundary.`

### T3: RESOLVED_WITH_FORMAL_NONCOMPARABILITY_BOUNDARY

**Reviewer strategy:** 实现 CPCS/TransCal 类与 ATC 类基线或形式化不可比性

**Evidence files:**

- `outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/analysis_report.json`
- `outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/independent_validation_report.json`
- `outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/cpcs_style_certified_outcomes.csv`
- `outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/atc_style_accuracy_estimation.csv`

**Manuscript anchors:**

- `The configuration-level CPCS-style candidate used source labels and audit-side unlabeled target geometry`
- `The ATC-style estimator produced absolute target-accuracy errors`
- `TransCal was not instantiated because its primary formulation calibrates logits from a target-adapted task classifier`

### T4: RESOLVED_AS_MEASUREMENT_ERROR_SENSITIVITY

**Reviewer strategy:** bootstrap relabeling、LCB endpoint 或阈值邻域排除敏感性

**Evidence files:**

- `outputs/dcs_opct_v11_probabilistic_event_risk_20260914/analysis_report.json`
- `outputs/dcs_opct_v11_probabilistic_event_risk_20260914/independent_validation_report.json`
- `outputs/dcs_opct_v11_target_action_endpoint_uncertainty_20260914/analysis_report.json`
- `outputs/dcs_opct_v11_target_action_endpoint_uncertainty_20260914/independent_validation_report.json`

**Manuscript anchors:**

- `We separately propagated endpoint measurement error by resampling test observations with crossed participant and physical-stimulus multinomial weights`
- `The frozen event is therefore retained for protocol continuity, not treated as a precisely observed biological state.`

### T5: PARTIALLY_RESOLVED_PROSPECTIVE_ATTEMPT_STRUCTURALLY_INELIGIBLE

**Reviewer strategy:** 进一步限定 claim；有条件时执行新的独立外部 effective release

**Evidence files:**

- `outputs/dcs_opct_v11_reviewer_closure_20260914/validation_report.json`
- `outputs/dcs_opct_v11_release_readiness/report.json`
- `outputs/amigos_v11_preaccess/validation_report.json`
- `outputs/emognition_v11_preaccess/validation_report.json`
- `docs/ekmed_v11_external_confirmation_implementation_lock.json`
- `outputs/ekmed_v11_external_confirmation/failure_001_structural_ineligibility.json`
- `outputs/ekmed_v11_external_confirmation/independent_validation_report.json`
- `docs/dcs_opct_v11_claim_evidence_matrix.md`

**Manuscript anchors:**

- `A post-hoc seven-domain refit retained positive mechanism directions and improved pooled probability error, but it failed its prespecified leave-dataset-out discrimination and operating-point gate.`
- `The EKM-ED attempt failed one stage earlier under a stronger prospective information boundary.`
- `EKM-ED met the prospective reservation, checksum, header-only schema, implementation-lock, and no-retuning requirements, but failed the frozen trial-completeness and participant-eligibility gate before model fitting.`
- `They do not establish a positive fully training-data-independent release, universal transport safety, successful prospective external effectiveness`

### T6: RESOLVED

**Reviewer strategy:** 增加数据访问边界和 transductive 标记

**Evidence files:**

- `docs/elsarticle/figures/fig_v11_three_stage_framework.pdf`
- `docs/elsarticle/figures/fig_v11_independence_ladder.pdf`
- `docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex`

**Manuscript anchors:**

- `Three-stage DCS-OPCT evidence chain and information-access boundaries.`
- `The primary analysis is transductive because the complete unlabeled target batch may enter transformation fitting`

### T7: RESOLVED

**Reviewer strategy:** 增加 decision level、target information、assumptions 或移至补充

**Evidence files:**

- `docs/dcs_opct_v11_closest_method_capability_matrix.md`
- `docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex`
- `outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/analysis_report.json`

**Manuscript anchors:**

- `Neutral resource and decision-level positioning of DCS-OPCT relative to neighboring method families.`
- `Rows describe primary formulations rather than every possible extension`

### T8: RESOLVED_PUBLIC_REPOSITORY_VERIFIED

**Reviewer strategy:** 发布版本化匿名复现包

**Evidence files:**

- `docs/dcs_opct_v11_public_repository_receipt_20260916.json`
- `docs/elsarticle/dcs_opct_v11_author_confirmation_checklist.md`
- `outputs/dcs_opct_v11_submission_artifacts_crossfit_v22_zip_verification.json`
- `scripts/finalize_dcs_opct_repository_link.py`

**Manuscript anchors:**

- `A checksum-verified reproducibility package, source code, aggregate outputs, and validation reports are publicly available`

### T9: RESOLVED_FROM_OFFICIAL_PROJECT_RECORDS

**Reviewer strategy:** 确认资助信息并加入正式 section

**Evidence files:**

- `docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex`
- `docs/dcs_opct_v11_eppvr_metadata_provenance_20260916.md`
- `docs/elsarticle/dcs_opct_v11_author_confirmation_checklist.md`

**Manuscript anchors:**

- `This work was supported by the National Natural Science Foundation of China (No. 62172081).`

### T10: RESOLVED

**Reviewer strategy:** 统一 outcome-held-out 术语并减少摘要 comparator 数字

**Evidence files:**

- `docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex`
- `docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex`
- `docs/elsarticle/dcs_opct_v11_bspc_highlights.txt`
- `docs/elsarticle/dcs_opct_v11_bspc_cover_letter.md`
- `outputs/dcs_opct_v11_reviewer_closure_20260914/validation_report.json`

**Manuscript anchors:**

- `Positive transport evidence remains retrospective and conditional on the evaluated target batch`
- `The study does not establish prospective external effectiveness`

### T11: RESOLVED_UNAVAILABLE_TRANSPARENTLY_DISCLOSED

**Reviewer strategy:** 从采集配置恢复最小数据字典；无法恢复则声明 unavailable

**Evidence files:**

- `docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex`
- `docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex`
- `docs/elsarticle/dcs_opct_v11_author_confirmation_checklist.md`

**Manuscript anchors:**

- `Channels 5--9 collectively comprise two EOG channels, one PPG channel, one GSR channel, and one temperature channel;`
- `their exact within-block order could not be recovered from the inspected acquisition and analysis records`
- `Neither channels 5--9 nor dominance enters the present analysis.`

## Submission decision boundary

- T8 is resolved by the verified public author-identifiable GitHub repository; T9 is resolved from official project records.
- T11 is resolved by explicitly recording that the exact within-block order is unavailable; the
  collective composition is disclosed and the channels are outside the executed analysis.
- T5 now includes a completed negative prospective EKM-ED endpoint-transport test, but remains the
  decisive scientific blocker for a positive prospective external-effectiveness claim. The bounded
  BSPC manuscript can be submitted without claiming that result.
- No task closure upgrades identity fallback into effectiveness, universal safety, or future-domain
  non-harm.
