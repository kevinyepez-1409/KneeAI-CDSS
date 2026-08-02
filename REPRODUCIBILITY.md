# Reproducibility and Artifact Provenance

This document describes the roles of the principal KneeAI artifacts and the limits of the available reproducibility evidence.

## Scope

KneeAI is a retrospective model-development and internal-validation research prototype for knee osteoarthritis severity assessment from radiographs.

The analyzed dataset contained:

- 8,260 radiographs
- 4,130 filename-derived grouping identifiers (FDGIs)
- 5,282 training images
- 1,322 tuning images
- 1,656 held-out internal test images

An FDGI is a filename-derived laterality-pair grouping proxy. It is not a verified patient, encounter, or longitudinal examination identifier.

## Canonical Historical Prediction Record

The historical fixed-prediction metrics reported in the manuscript are reproduced from:

```text
koa_5class_final_oversampled.npz
```

SHA-256:

```text
72dc11405b7f26de48426547ae1ba4882ae4a0347f1d236a862a44cb0d5d78ea
```

The corresponding CSV export is:

```text
loaded_predictions_with_patient_ids.csv
```

SHA-256:

```text
34df72e1b8fb1df2847dc37d746ed201ef1e5089b349cc1833fe6004c7df8881
```

The legacy column name `patient_id` stores the FDGI proxy and must not be interpreted as a verified clinical patient identifier.

The canonical historical fixed-prediction metrics are:

| Metric | Value |
|---|---:|
| Accuracy | 0.8219 |
| Balanced accuracy | 0.8544 |
| Macro F1-score | 0.8027 |
| Weighted F1-score | 0.8220 |
| Quadratic Cohen’s kappa | 0.7254 |
| Macro AUC | 0.9345 |
| Weighted AUC | 0.9088 |

These values characterize one archived prediction artifact and are not the principal matched three-seed average.

## Principal Matched Three-Seed Comparison

The primary formulation comparison used matched retraining with seeds 42, 123, and 2026.

| Metric | Hybrid KL 5-to-3 | Direct 3-class |
|---|---:|---:|
| Accuracy | 0.8114 ± 0.0082 | 0.8118 ± 0.0027 |
| Balanced accuracy | 0.7869 ± 0.0348 | 0.7616 ± 0.0281 |
| Macro F1-score | 0.7891 ± 0.0185 | 0.7801 ± 0.0146 |
| Quadratic Cohen’s kappa | 0.6937 ± 0.0188 | 0.6901 ± 0.0091 |
| Macro AUC | 0.9345 ± 0.0031 | 0.9303 ± 0.0012 |
| Weighted AUC | 0.9097 ± 0.0018 | 0.9040 ± 0.0016 |

All paired seed-level intervals for the principal metrics included zero. No general superiority, equivalence, or noninferiority was established.

## Historical Checkpoint

The checkpoint identified by the archived training notebook as the intended historical source is:

```text
efficientnetb3_5class_refined_v2.weights.h5
```

SHA-256:

```text
f69749315de3054c5925dbaf4cf411d7305a9d0d0c15bfdce1b4a4098c3ace49
```

The surviving H5 loads successfully with the reconstructed architecture. However, re-inference did not reproduce the archived NPZ probability matrix exactly. Therefore:

- the H5 is retained as the intended historical checkpoint;
- the NPZ/CSV pair remains the canonical source for the historical fixed-prediction metrics;
- the H5 must not be described as an exact numerical reproduction of the historical NPZ.

## Streamlit Deployment-Reference Checkpoint

The Streamlit application uses a separate deployment-reference checkpoint:

```text
kneeai_weights_final.weights.h5
```

SHA-256:

```text
49abd3fa257833176a4055f9f2c1a19169bd5e31dbc85f0067aef88399b49b5e
```

Preferred local path:

```text
models/kneeai_weights_final.weights.h5
```

This checkpoint is not the source of the archived historical accuracy of 0.8219.

The application verifies the SHA-256 digest before loading the checkpoint.

## KL 5-to-3 Rules

### Rule A — categorical decision

```text
five-class argmax
        ↓
KL-0 or KL-1 → Non-OA
KL-2 or KL-3 → Mild–Moderate OA
KL-4         → Severe OA
```

Rule A is used for the categorical output shown by the Streamlit application and for the historical categorical evaluation.

### Rule B — probability aggregation

```text
[
  p(KL-0) + p(KL-1),
  p(KL-2) + p(KL-3),
  p(KL-4)
]
```

Rule B is used for:

- aggregated probability summaries;
- one-vs-rest AUC;
- calibration diagnostics;
- normalized entropy;
- the aggregated Grad-CAM target.

Rule A and Rule B are related but not always equivalent.

## Preprocessing and Historical Training State

The archived inference workflow used:

- input size: 300 × 300 pixels;
- EfficientNet preprocessing;
- deterministic non-shuffled evaluation order;
- the default Keras image-generator interpolation behavior.

The effective historical fine-tuning state was the upper 50 backbone layers plus the classification head. A later code block was labeled as extending fine-tuning to 100 layers, but it did not re-enable the additional layers.

## Entropy Threshold

The Streamlit prototype uses:

```text
H = 0.60
```

This threshold is:

- post hoc;
- illustrative;
- selected after inspection of the internal test risk–coverage curve;
- not a prespecified or externally validated clinical operating point.

It must not be interpreted as a transferable reject threshold or clinical referral rule.

## Image-Level Duplicate Audit

The final image audit found no:

- binary-exact cross-subset duplicates;
- decoded-pixel-exact cross-subset duplicates;
- SSIM-confirmed cross-subset near-duplicates.

This reduces exact-image leakage risk but does not establish verified participant-level independence.

## Mendeley Data

Large model files, historical predictions, manifests, audit outputs, notebooks, scripts, and SHA-256 records are hosted in Mendeley Data:

```text
10.17632/cgjjbw8hsf
```

The original radiographs are not redistributed in this repository or in the reproducibility package.

## Reproduction Limits

The repository and Mendeley package support artifact traceability and partial reproduction of the reported analyses. They do not establish:

- exact recovery of the historical NPZ from the surviving historical H5;
- verified patient-level independence;
- external clinical validity;
- prospective workflow performance;
- medical-device readiness.

## Research-Use Disclaimer

KneeAI is for academic and research use only. It is not intended for diagnosis, treatment selection, autonomous triage, referral decisions, or patient management.
