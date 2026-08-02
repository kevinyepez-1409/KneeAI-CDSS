# KneeAI-CDSS

![DOI](https://img.shields.io/badge/DOI-10.17632%2Fcgjjbw8hsf-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Interface](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## KneeAI: Uncertainty-Aware Research Prototype for Knee Osteoarthritis Severity Assessment

KneeAI is a web-based research prototype for knee osteoarthritis (KOA) severity assessment from knee radiographs. It demonstrates an uncertainty-aware workflow that combines EfficientNetB3-based inference, coarse-from-fine Kellgren–Lawrence (KL) label consolidation, entropy-based ambiguity flagging, and Grad-CAM visual plausibility support.

This repository is associated with the manuscript:

**“Internal Validation of an Uncertainty-Aware Coarse-from-Fine Deep Learning Framework for Knee Osteoarthritis Severity Assessment.”**

> **Important notice:** KneeAI is a research prototype. It is not a clinically validated diagnostic device and must not be used for diagnosis, treatment selection, autonomous triage, referral decisions, or patient management. Interface messages are illustrative and institution-configurable.

---

## Study Scope

This work is a retrospective model-development and internal-validation study based on a publicly available radiographic dataset.

The analyzed dataset contained:

- **8,260 radiographs**
- **4,130 filename-derived grouping identifiers (FDGIs)**
- **5,282 training images**
- **1,322 tuning images**
- **1,656 held-out internal test images**

An **FDGI** is a filename-derived laterality-pair grouping proxy created from the numeric filename stem. It is **not** a verified patient, encounter, or longitudinal examination identifier.

The study does not establish:

- verified patient-level independence;
- external generalizability;
- prospective clinical effectiveness;
- regulatory readiness;
- a validated clinical operating threshold.

---

## Overview

KneeAI demonstrates how fine-grained KL supervision can be combined with a broader three-category output and uncertainty information in a research-prototype workflow.

The system integrates:

- an EfficientNetB3 backbone;
- fine-grained KL 0–4 supervision;
- a predefined KL 5-to-3 severity mapping;
- aggregated three-class probabilities;
- normalized Shannon entropy;
- Grad-CAM visual plausibility support;
- a Streamlit interface for research demonstration.

The three computational severity categories are:

- **Non-OA**
- **Mild–Moderate OA**
- **Severe OA**

These categories are research output strata and do not constitute a treatment guideline, referral rule, or validated clinical pathway.

---

## Hybrid KL 5-to-3 Strategy

The model produces five internal Softmax probabilities:

```text
[KL-0, KL-1, KL-2, KL-3, KL-4]
```

Two related but distinct operations are used.

### Rule A — categorical decision

The displayed categorical prediction follows the historical manuscript rule:

```text
five-class argmax
        ↓
KL-0 or KL-1 → Non-OA
KL-2 or KL-3 → Mild–Moderate OA
KL-4         → Severe OA
```

This rule is used for the mapped severity category shown by the application.

### Rule B — probability aggregation

For probability-level analyses, the five probabilities are aggregated as:

```text
[
  p(KL-0) + p(KL-1),
  p(KL-2) + p(KL-3),
  p(KL-4)
]
```

The aggregated vector is used for:

- probability summaries;
- one-vs-rest AUC;
- calibration diagnostics;
- normalized entropy;
- the aggregated-class Grad-CAM target.

Rule A and Rule B are intentionally reported separately because they are not always equivalent.

---

## Principal Matched Three-Seed Comparison

The principal formulation comparison used matched retraining with seeds **42, 123, and 2026** under the same FDGI manifest, training rows, preprocessing, augmentation, architecture apart from output dimension, optimization schedule, and tuning-subset-only checkpoint-selection rule.

| Metric | Hybrid KL 5-to-3 | Direct 3-class |
|---|---:|---:|
| Accuracy | 0.8114 ± 0.0082 | 0.8118 ± 0.0027 |
| Balanced accuracy | 0.7869 ± 0.0348 | 0.7616 ± 0.0281 |
| Macro F1-score | 0.7891 ± 0.0185 | 0.7801 ± 0.0146 |
| Quadratic Cohen’s kappa | 0.6937 ± 0.0188 | 0.6901 ± 0.0091 |
| Macro AUC | 0.9345 ± 0.0031 | 0.9303 ± 0.0012 |
| Weighted AUC | 0.9097 ± 0.0018 | 0.9040 ± 0.0016 |

All paired seed-level intervals for the principal performance metrics included zero. Therefore, **no general superiority, equivalence, or noninferiority was established**.

---

## Secondary Historical Fixed-Prediction Result

A validated historical hybrid prediction artifact produced the following internal held-out test results:

| Metric | Value |
|---|---:|
| Accuracy | 0.8219 |
| Balanced accuracy | 0.8544 |
| Macro F1-score | 0.8027 |
| Weighted F1-score | 0.8220 |
| Quadratic Cohen’s kappa | 0.7254 |
| Macro AUC | 0.9345 |
| Weighted AUC | 0.9088 |
| Severe OA recall | 0.9412 |

These values characterize one validated historical prediction artifact and are **not** the principal three-seed average.

The historical metrics are reproduced from the canonical archived prediction files:

```text
koa_5class_final_oversampled.npz
loaded_predictions_with_patient_ids.csv
```

The legacy column name `patient_id` in the CSV stores the FDGI proxy and does not represent a verified clinical patient identifier.

---

## Historical and Deployment Artifact Roles

The revised reproducibility package distinguishes three artifact roles.

### Canonical historical prediction record

```text
koa_5class_final_oversampled.npz
```

SHA-256:

```text
72dc11405b7f26de48426547ae1ba4882ae4a0347f1d236a862a44cb0d5d78ea
```

### Intended historical checkpoint

```text
efficientnetb3_5class_refined_v2.weights.h5
```

SHA-256:

```text
f69749315de3054c5925dbaf4cf411d7305a9d0d0c15bfdce1b4a4098c3ace49
```

The archived training notebook identifies this H5 as the checkpoint loaded before export of the historical predictions. However, re-inference from the surviving H5 did not reproduce the archived probability matrix exactly. It is therefore retained with a reproducibility warning and is not treated as the canonical source of the historical metrics.

### Streamlit deployment-reference checkpoint

```text
kneeai_weights_final.weights.h5
```

SHA-256:

```text
49abd3fa257833176a4055f9f2c1a19169bd5e31dbc85f0067aef88399b49b5e
```

This is the checkpoint used by the Streamlit application. It is a separate deployment-reference artifact and is **not** presented as the source of the historical accuracy of 0.8219.

---

## Uncertainty-Aware Ambiguity Flagging

KneeAI calculates normalized Shannon entropy from the aggregated three-class probability vector.

The application uses:

```text
H = 0.60
```

only as an illustrative prototype setting.

- **H ≤ 0.60:** the output does not exceed the illustrative threshold.
- **H > 0.60:** the output is flagged as higher entropy in the research prototype.

This threshold was selected post hoc after inspection of the held-out internal test risk–coverage curve. It was not prespecified, selected on independent tuning data, or externally validated. It must not be interpreted as a transferable clinical threshold or reject option.

---

## Model Architecture

- **Backbone:** EfficientNetB3
- **Input size:** 300 × 300 pixels
- **Internal output:** five KL-grade probabilities
- **Mapped output:** three computational severity categories
- **Optimization:** Optuna with a Tree-structured Parzen Estimator sampler
- **Regularization:** dropout, L2 regularization, and label smoothing
- **Historical effective fine-tuning state:** upper 50 backbone layers and classification head
- **Visual plausibility:** Grad-CAM
- **Uncertainty summary:** normalized Shannon entropy

A later historical code block was labeled as extending fine-tuning to 100 backbone layers, but it did not re-enable the additional layers. The effective historical state therefore remained the upper 50 backbone layers.

---

## Web Application

The Streamlit application allows users to:

- upload a knee radiograph;
- obtain the internal five-class probability profile;
- obtain the mapped three-category severity output using Rule A;
- inspect the aggregated three-class probability profile;
- view normalized entropy;
- view a Grad-CAM visual plausibility map.

The application verifies the deployment checkpoint using SHA-256 before loading it.

Grad-CAM is included only as a qualitative visual plausibility tool. It is not causal evidence of model reasoning and does not replace expert review.

---

## Repository Structure

```text
KneeAI-CDSS/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── REPRODUCIBILITY.md
├── CITATION.cff
│
├── models/
│   └── README.md
│
├── docs/
│
├── demo/
│
└── notebooks/
    └── README.md
```

Large reproducibility artifacts and model files are hosted externally in Mendeley Data rather than committed to this repository.

---

## Model Weights and Reproducibility Package

The dataset used in the study is publicly available from the **Knee Osteoarthritis Dataset with Severity Grading** repository on Kaggle.

The revised reproducibility package is available through Mendeley Data:

**DOI:** `10.17632/cgjjbw8hsf`

The package includes:

- the canonical historical NPZ prediction artifact;
- the FDGI-linked historical CSV export;
- six tuning-selected matched-retraining checkpoints;
- the intended historical checkpoint with a reproducibility warning;
- the separate deployment-reference checkpoint;
- the archived direct three-class comparator;
- split manifests;
- exact matched-training rows;
- per-run histories and predictions;
- calibration, bootstrap, checkpoint-selection, and duplicate-audit outputs;
- SHA-256 manifests;
- notebooks and supporting scripts.

The original radiographs are not redistributed.

---

## Installing the Deployment Checkpoint

Download this exact file from the Mendeley Data package:

```text
kneeai_weights_final.weights.h5
```

Place it in:

```text
models/kneeai_weights_final.weights.h5
```

Verify its SHA-256:

```text
49abd3fa257833176a4055f9f2c1a19169bd5e31dbc85f0067aef88399b49b5e
```

The application also supports a legacy root-level location for backward compatibility, but `models/` is the preferred documented path.

Do not rename another checkpoint to this filename.

---

## Running the App

Install the application dependencies:

```bash
python -m pip install -r requirements.txt
```

Run Streamlit:

```bash
python -m streamlit run app.py
```

The application will refuse to load the model when the checkpoint SHA-256 does not match the expected deployment-reference hash.

Optional environment variables:

```text
KNEEAI_MODEL_PATH
KNEEAI_EXPECTED_SHA256
```

`KNEEAI_MODEL_PATH` may be used to point to an exact local checkpoint path. `KNEEAI_EXPECTED_SHA256` may be used only when intentionally testing another explicitly documented artifact.

---

## Limitations

KneeAI should be interpreted as a research prototype only.

Important limitations include:

- single public dataset mirror;
- no verified patient, encounter, or longitudinal identifiers;
- no external multicenter validation;
- no prospective workflow evaluation;
- no reader study;
- no demographic or acquisition-domain generalizability analysis;
- only three matched training seeds;
- limited held-out Severe OA support;
- post-hoc entropy threshold;
- qualitative Grad-CAM evidence;
- inconclusive auxiliary SHAP findings;
- no validated medical-device or clinical-use status.

The image-level audit found no binary-exact, decoded-pixel-exact, or SSIM-confirmed near-duplicate images across the final subsets. This reduces exact-image leakage risk but does not establish verified participant-level independence.

---

## Citation

When using this repository, cite the associated manuscript and Mendeley Data record.

```bibtex
@misc{kneeai_cdss_2026,
  title        = {KneeAI-CDSS: Uncertainty-Aware Research Prototype for Knee Osteoarthritis Severity Assessment},
  author       = {Yepez, Kevin Alejandro and Villacreses, Emmily},
  year         = {2026},
  publisher    = {Mendeley Data},
  doi          = {10.17632/cgjjbw8hsf},
  url          = {https://doi.org/10.17632/cgjjbw8hsf}
}
```

Associated manuscript:

```text
Internal Validation of an Uncertainty-Aware Coarse-from-Fine
Deep Learning Framework for Knee Osteoarthritis Severity Assessment
```

---

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.

---

## Disclaimer

This software is provided exclusively for academic and research purposes. It is not intended for clinical diagnosis, treatment selection, autonomous triage, referral decisions, or medical decision-making. Clinical use would require verified clinical identifiers, external validation, prospective evaluation, reader studies, institutional approval, regulatory review, and compliance with applicable medical-device requirements.
