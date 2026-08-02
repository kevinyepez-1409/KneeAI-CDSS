# Model Weights

Large model files are hosted in the associated Mendeley Data reproducibility package rather than committed to GitHub.

## Mendeley Data

```text
10.17632/cgjjbw8hsf
```

## Deployment Checkpoint Used by the Streamlit App

Download this exact file:

```text
kneeai_weights_final.weights.h5
```

Place it at:

```text
models/kneeai_weights_final.weights.h5
```

Expected SHA-256:

```text
49abd3fa257833176a4055f9f2c1a19169bd5e31dbc85f0067aef88399b49b5e
```

The application verifies this digest before loading the checkpoint. Do not rename another H5 file to this filename.

This checkpoint is a separate deployment-reference artifact. It is **not** the source of the archived historical accuracy of 0.8219.

## Historical Fixed-Prediction Artifact

The historical fixed-prediction metrics are reproduced from:

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

The legacy `patient_id` column stores the filename-derived grouping identifier (FDGI) proxy and does not represent a verified clinical patient identifier.

## Intended Historical Checkpoint

The archived training notebook identifies:

```text
efficientnetb3_5class_refined_v2.weights.h5
```

SHA-256:

```text
f69749315de3054c5925dbaf4cf411d7305a9d0d0c15bfdce1b4a4098c3ace49
```

as the intended historical checkpoint loaded before prediction export.

The surviving H5 loads successfully with the reconstructed architecture, but re-inference did not reproduce the archived NPZ probability matrix exactly. Therefore:

- the H5 is retained as the intended historical checkpoint;
- the NPZ/CSV pair remains the canonical source of the historical fixed-prediction metrics;
- the H5 must not be described as an exact numerical reproduction of the historical NPZ.

## Other Checkpoints

The Mendeley Data package also contains the matched-retraining checkpoints and the archived direct three-class comparator used in the manuscript analyses.

These files are retained for artifact traceability and should not be substituted for the deployment checkpoint unless the application code and expected SHA-256 are intentionally changed.

## Local Verification

From the repository root, verify the deployment checkpoint with PowerShell:

```powershell
Get-FileHash ".\models\kneeai_weights_final.weights.h5" -Algorithm SHA256
```

Expected result:

```text
49ABD3FA257833176A4055F9F2C1A19169BD5E31DBC85F0067AEF88399B49B5E
```

Then run:

```powershell
python -m streamlit run app.py
```

## Important Notice

These weights are provided for academic and research use only. They are not clinically validated and must not be used for diagnosis, treatment selection, autonomous triage, referral decisions, or patient management.
