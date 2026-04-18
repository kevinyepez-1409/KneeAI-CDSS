# KneeAI-CDSS
![DOI](https://img.shields.io/badge/DOI-10.17632%2Fcgjjbw8hsf.1-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Interface](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
## 🩺 Uncertainty-Aware Clinical Decision Support System for Knee Osteoarthritis

KneeAI is a web-based Clinical Decision Support System (CDSS) for the automatic severity classification of knee osteoarthritis (KOA) from radiographic images. The system was developed to improve clinical triage by combining deep learning, uncertainty quantification, and visual explainability in an accessible diagnostic support tool.

---

## 🧠 Clinical Overview

KneeAI was designed as a clinically oriented AI system rather than a conventional image classifier.  
Its objective is to support decision-making in musculoskeletal assessment by identifying radiographic severity patterns of knee osteoarthritis and presenting interpretable outputs for healthcare professionals.

The system integrates:

- Deep learning with EfficientNetB3
- Clinical label consolidation from 5 classes to 3 clinically meaningful categories
- Uncertainty quantification using Shannon entropy
- Visual explainability using Grad-CAM
- A web-based interface for real-time inference using Streamlit

The final clinical classes are:

- **Non-OA**
- **Mild-Moderate**
- **Severe**

---

## 🚨 Core Innovation: Uncertainty-Aware Decision Making

KneeAI incorporates a safety mechanism based on **normalized Shannon entropy** to identify ambiguous predictions.

- If **entropy < 0.6** → the prediction is accepted
- If **entropy ≥ 0.6** → the case is flagged for specialist review

This uncertainty gate prevents overconfident automatic predictions in clinically ambiguous cases and improves reliability in real-world settings.

---

## 📊 Performance Summary

| Metric | Value |
|------|------|
| Accuracy | 82.21% |
| AUC | 0.918 |
| Cohen’s Kappa | 0.7254 |
| Sensitivity (Severe) | >80% |
| Statistical Significance | p < 0.001 |

The proposed 3-class clinical formulation significantly outperformed the original 5-class formulation when both were evaluated under the same clinical label space.

---

## ⚙️ Model Architecture

- **Backbone:** EfficientNetB3
- **Input size:** 300 × 300 pixels
- **Output:** 5-class KL grading, collapsed into 3 clinical classes
- **Optimization:** Bayesian optimization with Optuna
- **Regularization:** Dropout + L2 regularization
- **Training strategy:** warm-up + progressive fine-tuning
- **Explainability:** Grad-CAM
- **Uncertainty module:** Shannon entropy thresholding

---

## 🧬 Clinical Label Mapping

The original Kellgren-Lawrence (KL) grading system was consolidated into a 3-class framework to improve robustness and clinical interpretability:

- **KL-0 → Non-OA**
- **KL-1 → Non-OA**
- **KL-2 → Mild-Moderate**
- **KL-3 → Mild-Moderate**
- **KL-4 → Severe**

This reformulation reduces ambiguity between adjacent intermediate grades and better aligns with practical clinical triage.

---

## 🖥️ Web Application

The system was implemented as a Streamlit-based web application to allow healthcare professionals to:

- Upload AP knee radiographs
- Obtain automatic severity classification
- Visualize confidence and uncertainty
- Review Grad-CAM heatmaps for anatomical explainability

This allows direct interaction with the model without requiring programming knowledge or specialized deployment environments.

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
├── src/
│   ├── model.py
│   ├── inference.py
│   ├── explainability.py
│   └── utils.py
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
