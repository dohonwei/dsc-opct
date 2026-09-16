# EPPVR Metadata Provenance Record

Date: 2026-09-16

This record documents the source basis for EPPVR funding, ethics wording, and
channel metadata used in the DCS-OPCT BSPC submission. It separates facts that
can be verified from archived source records from details that remain
unrecoverable. No metadata were inferred from directory names alone.

## Funding

The official National Natural Science Foundation of China project plan and
approval notice both identify project No. 62172081, titled "Key technologies
and theoretical research on emotion physiological sensing HMD interaction
interfaces in virtual environments" (English translation for documentation;
the official records are in Chinese). The project leader is Dongyi Chen. The
previous EPPVR study using the same acquisition platform and ethics approval
also acknowledges the National Natural Science Foundation of China under No.
62172081.

The BSPC manuscript therefore uses the bounded statement:

> This work was supported by the National Natural Science Foundation of China
> (No. 62172081).

The prior EPPVR paper additionally lists two Guangxi grants. They are not
carried into the current manuscript because no inspected record establishes
that they funded the present DCS-OPCT study.

## Ethics wording

The previous EPPVR paper prints the following institutional wording for the
same approval number, 106142023122227999:

> Ethical Committee of the University of Electronic Science and Technology of
> China

The manuscript and supplement use this wording consistently. Written informed
consent and secondary-analysis authorization remain based on the author's
study records already supplied for this project.

## Channel indexing

The archived analysis code verifies the processed five-channel subset as:

- index 0: FP1;
- index 1: FP2;
- index 2: GSR/EDA;
- index 3: PPG/BVP;
- index 4: SKT/HST.

The available code does not read channels 5--9 and does not define their exact
within-block order. Author-supplied records establish only their collective
composition: two EOG channels, one PPG channel, one GSR channel, and one
temperature channel. The submission therefore reports the collective
composition and explicitly declines to guess the unresolved index order.

## Source integrity

| Source record | SHA-256 | Evidential use |
|---|---|---|
| `Emotion recognition based on a limited number of multimodal physiological signals channels.pdf` | `694E2ADE24ACC83044CDD3B8751369DECF7D80105EB7006E0056863AF9B9451E` | Same EPPVR platform; printed funding and ethics wording |
| `国家自然科学基金资助项目计划书.pdf` | `B156F207EADF8B60D67CBFA52448AD361A2EA2868E532F481B66971A8C8AA304` | Official project number, title, and project leader |
| `国家自然科学基金资助项目批准通知.pdf` | `05BED2CB37844DCA653122F7E6AD9455B87A9372C2AE159D0844E85BD732884A` | Independent official confirmation of project No. 62172081 |
| `feature_extract_hmd.py` | `BD641526A680AFFB8CF7393A40DED894A37F3728EBDC7406022311AD0F4780CD` | Processed channel indices 0--4 |
| `emotion_classify.py` | `9522A0F2BC9DEE68C7E6818CE1B4650886878738BD8BA9BD581AD3DF4E40AFB4` | Independent code confirmation of processed channel indices 0--4 |

The external source files remain in their governed local archives and are not
redistributed in the anonymous package. This provenance record contains only
the minimum metadata and checksums needed to audit the manuscript statements.
