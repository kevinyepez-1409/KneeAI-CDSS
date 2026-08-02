# Notebooks and Reproducibility

This directory provides documentation for the Jupyter notebooks and supporting scripts associated with the KneeAI study.

The full notebook collection, large outputs, model checkpoints, manifests, and audit files are hosted in the associated Mendeley Data reproducibility package rather than committed to GitHub.

## Mendeley Data

```text
10.17632/cgjjbw8hsf
```

## Scope of the Notebook Collection

The archived materials cover the principal stages of the study:

1. dataset inspection and KL-label consolidation;
2. EfficientNetB3 five-class development;
3. direct three-class comparison;
4. filename-derived grouping identifier (FDGI) split preparation;
5. hyperparameter optimization;
6. staged transfer learning and checkpoint selection;
7. matched three-seed hybrid-versus-direct retraining;
8. held-out internal evaluation;
9. calibration and uncertainty analysis;
10. Grad-CAM visual plausibility analysis;
11. auxiliary SHAP analysis;
12. architecture and classical machine-learning benchmarks;
13. ablation studies;
14. bootstrap and paired statistical analyses;
15. checkpoint, prediction, duplicate, and provenance audits.

The notebook collection is therefore broader than a three-notebook workflow.

## Key Historical Notebook Roles

The archived project includes notebooks with roles such as:

### KL grading and formulation comparison

Documents the relationship between the original KL 0–4 labels and the predefined three-category output:

```text
KL-0 or KL-1 → Non-OA
KL-2 or KL-3 → Mild–Moderate OA
KL-4         → Severe OA
```

### Five-class EfficientNetB3 development

Documents the historical five-class architecture, preprocessing, augmentation, optimization, and staged fine-tuning workflow.

### FDGI-aware split and evaluation workflow

Documents the train, tuning, and held-out internal test partitions using filename-derived grouping identifiers.

An FDGI is a filename-derived laterality-pair grouping proxy. It is not a verified patient, encounter, or longitudinal examination identifier.

### Matched hybrid-versus-direct retraining

Documents the controlled comparison using seeds:

```text
42, 123, 2026
```

The principal comparison used matched training rows, the same FDGI manifest, the same optimization schedule, and tuning-subset-only checkpoint selection.

### Calibration and uncertainty

Documents aggregated three-class probabilities, calibration diagnostics, normalized Shannon entropy, and risk–coverage analysis.

The prototype threshold:

```text
H = 0.60
```

is post hoc and illustrative. It is not a validated clinical operating threshold.

### Grad-CAM and auxiliary interpretation analyses

Documents Grad-CAM visual plausibility outputs and auxiliary SHAP analyses.

Grad-CAM is not causal evidence of model reasoning, and the SHAP analyses are considered exploratory and auxiliary.

### Reproducibility and provenance audits

Documents:

- checkpoint inventories;
- SHA-256 verification;
- prediction-artifact comparison;
- FDGI order and laterality checks;
- cross-subset duplicate analysis;
- historical H5-to-NPZ reproduction attempts;
- packaging and provenance review.

## Historical Prediction Source

The historical fixed-prediction metrics are reproduced from the canonical archived NPZ:

```text
koa_5class_final_oversampled.npz
```

SHA-256:

```text
72dc11405b7f26de48426547ae1ba4882ae4a0347f1d236a862a44cb0d5d78ea
```

The associated CSV export is:

```text
loaded_predictions_with_patient_ids.csv
```

SHA-256:

```text
34df72e1b8fb1df2847dc37d746ed201ef1e5089b349cc1833fe6004c7df8881
```

The legacy `patient_id` column stores the FDGI proxy and must not be interpreted as a verified clinical patient identifier.

## Historical Checkpoint Warning

The intended historical checkpoint is:

```text
efficientnetb3_5class_refined_v2.weights.h5
```

SHA-256:

```text
f69749315de3054c5925dbaf4cf411d7305a9d0d0c15bfdce1b4a4098c3ace49
```

The surviving H5 loads successfully with the reconstructed architecture, but re-inference did not reproduce the archived NPZ probability matrix exactly.

Therefore:

- the NPZ/CSV pair is the canonical source for the historical fixed-prediction metrics;
- the H5 is retained as the intended historical checkpoint;
- the notebook archive supports artifact traceability but not exact recovery of the historical NPZ from that H5.

## Deployment Checkpoint

The Streamlit application uses a separate checkpoint:

```text
kneeai_weights_final.weights.h5
```

SHA-256:

```text
49abd3fa257833176a4055f9f2c1a19169bd5e31dbc85f0067aef88399b49b5e
```

This deployment-reference checkpoint is not the source of the archived historical accuracy of 0.8219.

## Rule A and Rule B

The archived workflow distinguishes two related operations.

### Rule A — categorical output

```text
five-class argmax
        ↓
predefined KL 5-to-3 mapping
```

Rule A is used for the historical categorical evaluation and the mapped category displayed by the Streamlit application.

### Rule B — probability aggregation

```text
[
  p(KL-0) + p(KL-1),
  p(KL-2) + p(KL-3),
  p(KL-4)
]
```

Rule B is used for:

- probability summaries;
- one-vs-rest AUC;
- calibration;
- normalized entropy;
- aggregated Grad-CAM targets.

The two rules are not always equivalent.

## Historical Training-State Note

The effective historical fine-tuning state was the upper 50 EfficientNetB3 backbone layers plus the classification head.

A later code block was labeled as extending fine-tuning to 100 backbone layers, but it did not re-enable the additional layers. The effective state therefore remained 50 trainable backbone layers.

## Running the Notebooks

The archived notebooks were developed with local absolute paths and may require path updates before execution.

Users should:

1. create an isolated Python environment;
2. install the versions documented in the archived environment and package metadata;
3. download the public radiographic dataset separately;
4. update local dataset and output paths;
5. preserve the provided FDGI split manifests;
6. verify downloaded artifacts using the included SHA-256 manifests;
7. avoid overwriting archived outputs when performing new runs.

The historical development environment included Python 3.10 and TensorFlow/Keras 2.10 with a DirectML-compatible setup. Exact hardware and software behavior may differ in newer environments.

## Execution Order

The notebooks are not guaranteed to form one strictly linear, end-to-end sequence.

Some files are:

- historical development notebooks;
- analysis-only notebooks;
- recovery or reconstruction scripts;
- audit notebooks;
- sensitivity analyses;
- manuscript-revision workflows.

Before execution, consult the package README, manifests, filenames, and output directories to determine the intended role of each file.

## Data Availability

The original radiographs are not redistributed in GitHub or in the Mendeley reproducibility package.

Users must obtain the source dataset separately and comply with its original terms of use.

## Reproducibility Statement

The Mendeley package is best described as a:

```text
reproducibility and artifact-traceability package
```

It supports inspection of:

- archived predictions;
- model checkpoints;
- matched-training outputs;
- manifests;
- statistical analyses;
- calibration and interpretation outputs;
- audit evidence;
- SHA-256 records.

It should not be described as proving exact numerical reproduction of every historical output from every surviving checkpoint.

## Research-Use Disclaimer

The notebooks, scripts, models, and outputs are provided for academic and research use only. They are not clinically validated and must not be used for diagnosis, treatment selection, autonomous triage, referral decisions, or patient management.
