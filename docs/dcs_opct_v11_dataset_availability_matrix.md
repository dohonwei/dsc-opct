# DCS-OPCT v11 dataset and reproducibility matrix

This matrix records the evidence role and release boundary of every dataset in
the DCS-OPCT v11 study. It is a reproducibility statement, not a legal opinion;
the original provider license, data-use agreement, and participant consent take
precedence over this table.

| Dataset | Current evidence role | Raw-data access | Shareable project artifact | Boundary |
|---|---|---|---|---|
| DEAP | Stage-I mechanism experiment and Stage-II source domain | Original provider application and conditions | Aggregate tables, configuration-level predictions, split identifiers, and code where permitted | No redistribution of provider raw EEG |
| MAHNOB-HCI | Stage-I mechanism experiment and Stage-II source domain | Original provider application and conditions | Aggregate tables, configuration-level predictions, normalized stimulus mapping audit, and code where permitted | Session media and raw signals remain provider controlled |
| EPPVR | Retrospective DCS-OPCT development domain; real two-frontal-channel wearable data | Author-controlled research dataset subject to consent and institutional approval | De-identified aggregate/configuration outputs and analysis code after governance review | One session per participant; no valid shared physical-stimulus identifier; no stimulus-axis or cross-session claim |
| CASE | Retrospective DCS-OPCT development domain | Original provider conditions | Aggregate/configuration outputs, frozen assignments, and code where permitted | Not independent confirmation after method development |
| CEAP-360VR | Retrospective DCS-OPCT development domain | Original provider conditions | Aggregate/configuration outputs, frozen assignments, and code where permitted | Earlier confirmation failure remains preserved; current use is development evidence |
| SEED-IV | Retrospective DCS-OPCT development domain after an earlier frozen negative test | Original provider conditions | Aggregate/configuration outputs, failure record, assignments, and code where permitted | Cannot be reused as untouched confirmation |
| DREAMER | Retrospective DCS-OPCT development domain | Original provider conditions | Aggregate/configuration outputs, frozen assignments, and code where permitted | Not independent confirmation after method development |
| AVDOS-VR | Post-access exploratory cross-physiology stress test | Original provider conditions | Aggregate PPG dose, action-lock, gate, and validation artifacts where permitted | Registered branch did not complete; identity fallback is non-intervention, not effectiveness |
| FACED | Post-access exploratory common-montage EEG boundary test | Original provider conditions | Aggregate dose, identity-probe, applicability, failure, and validation artifacts where permitted | Registered 32-channel cohort was ineligible; zero endpoint events make effectiveness non-estimable |
| AMIGOS | Prospectively reserved future one-shot EEG confirmation | Official EULA-authorized archive required; not present locally | Reservation, three official-route access records, schema-only inspector, synthetic tests, implementation-lock tooling, and 13/13 CUDA pre-access report | No participant value or outcome has been accessed; no external result exists |
| Emognition | Prospectively reserved future four-channel wearable-EEG replication | Written EULA authorization and official archive required; not present locally | Reservation, archived official EULA and metadata receipt, field- and archive-level fail-closed schema tests, staged GPU runner, implementation-lock tooling, and 15/15 CUDA pre-access report | No archive member or participant value has been accessed; no external result exists |

The public reproducibility package should include only artifacts allowed by the
corresponding source terms. When row-level derivatives cannot be redistributed,
the package retains deterministic scripts, checksums, schemas, assignments,
aggregate tables, and independent validation reports so an authorized user can
reconstruct the analysis locally.
