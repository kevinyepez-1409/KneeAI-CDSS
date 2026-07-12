# KneeAI-CDSS

![DOI](https://img.shields.io/badge/DOI-10.17632%2Fcgjjbw8hsf.1-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Interface](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🩺 KneeAI: Uncertainty-Aware Research Prototype for Knee Osteoarthritis Severity Assessment

KneeAI is a web-based research prototype for knee osteoarthritis (KOA) severity assessment from radiographic images. It demonstrates an uncertainty-aware workflow that combines deep learning, hybrid Kellgren–Lawrence (KL) label consolidation, entropy-based ambiguity flagging, and Grad-CAM visual plausibility support.

This repository is associated with the manuscript:

**“Uncertainty-Aware Clinical Decision Support for Knee Osteoarthritis Severity Assessment from Radiographs Using a Hybrid KL 5-to-3 Strategy.”**

> **Important notice:** KneeAI is a research prototype. It is not a clinically validated diagnostic device, and it should not be used as a substitute for professional clinical judgment. Any interface messages are illustrative and institution-configurable.

---

## 🧠 Overview

KneeAI was designed as a deployment-oriented research prototype rather than a conventional image classifier alone. Its objective is to demonstrate how radiographic KOA severity estimation can be combined with uncertainty filtering and visual plausibility outputs in a screening-oriented workflow.

The system integrates:

* Deep learning with an EfficientNetB3 backbone
* Fine-grained KL 0–4 supervision during training
* Hybrid KL 5-to-3 clinical label consolidation at inference
* Entropy-based ambiguity flagging
* Grad-CAM visual plausibility support
* A Streamlit-based web interface for research demonstration

The final deployment-level categories are:

* **Non-OA**
* **Mild–Moderate OA**
* **Severe OA**

---

## 🔄 Hybrid KL 5-to-3 Strategy

The original Kellgren–Lawrence grading system was used during training to preserve fine-grained radiographic supervision. At inference, the internal five-class probabilities were consolidated into three clinically oriented categories:

* **KL-0 → Non-OA**
* **KL-1 → Non-OA**
* **KL-2 → Mild–Moderate OA**
* **KL-3 → Mild–Moderate OA**
* **KL-4 → Severe OA**

This hybrid design separates representation learning from deployment-level decision output. The model learns from the original KL 0–4 structure, while the final output is simplified into broader categories intended for screening- and triage-oriented research support.

---

## 🚨 Uncertainty-Aware Ambiguity Flagging

KneeAI includes an entropy-based ambiguity module to identify uncertain predictions.

* If **H ≤ 0.6**, the case is treated as non-ambiguous within the research prototype.
* If **H > 0.6**, the case is flagged as ambiguous and routed for specialist review.

This threshold was internally evaluated using risk–coverage analysis. It should not be interpreted as a clinically validated decision threshold.

---

## 📊 Performance Summary

The selected hybrid EfficientNetB3 reference checkpoint was evaluated on an independent corrected-identifier-level test subset.

| Metric                            | Value  |
| --------------------------------- | ------ |
| Accuracy                          | 0.8219 |
| Balanced accuracy                 | 0.8544 |
| Macro AUC                         | 0.9345 |
| Weighted AUC                      | 0.9088 |
| Quadratic Cohen’s kappa           | 0.7254 |
| Macro F1-score                    | 0.8027 |
| Severe OA recall                  | 0.9412 |
| Ambiguous fraction at H > 0.6     | 0.3140 |
| Retained-case accuracy at H ≤ 0.6 | 0.8961 |

The hybrid KL 5-to-3 strategy showed stronger fixed-checkpoint performance than the direct 3-class EfficientNetB3 comparator on the same independent corrected-identifier-level test subset. These estimates are internal and conditioned on the selected checkpoint; they do not represent external clinical validation or matched retraining-variability-adjusted effects.

---

## ⚙️ Model Architecture

* **Backbone:** EfficientNetB3
* **Input size:** 300 × 300 pixels
* **Internal output:** Five KL-grade probabilities
* **Deployment-level output:** Three clinically consolidated categories
* **Optimization:** Bayesian optimization with Optuna
* **Regularization:** Dropout, L2 regularization, and label smoothing
* **Training strategy:** Warm-up followed by partial fine-tuning
* **Explainability:** Grad-CAM
* **Uncertainty module:** Normalized Shannon entropy-based ambiguity flagging

---

## 🖥️ Web Application

The Streamlit interface demonstrates how the model output can be presented in a research-prototype setting. The application allows users to:

* Upload an AP knee radiograph
* Obtain a deployment-level KOA severity category
* Visualize confidence and uncertainty information
* Review Grad-CAM visual plausibility heatmaps
* Identify ambiguous cases that should be reviewed by a specialist

The interface is intended for research demonstration only. The displayed output messages are illustrative placeholders and should be locally reviewed, replaced, or disabled before any prospective clinical workflow evaluation.

---

## 📂 Repository Structure

```bash
KneeAI-CDSS/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
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

Depending on the release version, trained weights and larger reproducibility artifacts may be hosted externally through the associated Mendeley Data repository rather than stored directly in GitHub.

---

## 📦 Model Weights and Data

The dataset used in the study is publicly available from the Knee Osteoarthritis Dataset with Severity Grading repository on Kaggle.

Supporting materials, reproducibility artifacts, and model-related files are available through the associated Mendeley Data repository:

**DOI:** `10.17632/cgjjbw8hsf.1`

If the model weights are not included directly in this GitHub repository, download the final weights from the data repository and place them in the repository root as:

```bash
kneeai_weights_final.weights.h5
```

---

## 🚀 Running the App

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The app expects the final weights file to be available at:

```bash
kneeai_weights_final.weights.h5
```

---

## ⚠️ Limitations

KneeAI should be interpreted as a research prototype only. The current results are based on internal corrected-identifier-level evaluation using a public radiographic dataset. The system has not undergone prospective clinical validation, external multicenter validation, regulatory review, or real-world workflow evaluation.

Important limitations include:

* No prospective clinical validation
* No external multicenter validation
* No demographic or acquisition-metadata generalizability assessment
* Limited Severe OA sample size in the independent test subset
* Entropy threshold internally evaluated only
* Grad-CAM used as visual plausibility support, not causal evidence of model reasoning
* Interface messages are illustrative and institution-configurable, not clinical recommendations

---

## 📚 Citation

If you use this repository or the associated data artifacts, please cite the corresponding manuscript and Mendeley Data repository.

```bibtex
@misc{kneeai_cdss,
  title        = {KneeAI-CDSS: Uncertainty-Aware Research Prototype for Knee Osteoarthritis Severity Assessment},
  author       = {Yepez, Kevin Alejandro and Villacreses, Emmily},
  year         = {2026},
  publisher    = {Mendeley Data},
  doi          = {10.17632/cgjjbw8hsf.4}
}
```

---

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for details.

---

## Disclaimer

This software is provided for academic and research purposes only. It is not intended for clinical diagnosis, treatment selection, autonomous triage, or medical decision-making. Clinical use would require external validation, prospective evaluation, institutional approval, and compliance with applicable medical-device regulations.
