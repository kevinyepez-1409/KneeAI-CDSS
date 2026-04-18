# Notebooks & Reproducibility

This folder provides a reference to the Jupyter notebooks that document the complete experimental pipeline for the KneeAI clinical decision support system.

Due to file size constraints and to ensure reproducibility, the full notebooks are hosted externally in Mendeley Data.

---

## 📦 Access to Notebooks

The complete experimental workflow is available at:

https://doi.org/10.17632/cgjjbw8hsf.1

---

## 🧠 Notebooks Overview

The project includes three main notebooks:

### 1. KL Grading Comparison (3-Class vs 5-Class)

* Comparative analysis between the traditional 5-class KL grading system and a clinically consolidated 3-class scheme
* Evaluation of class ambiguity in intermediate grades
* Justification for clinical label simplification

---

### 2. EfficientNetB3 Baseline Training (5-Class)

* Data preprocessing and augmentation
* Model training using EfficientNetB3
* Performance evaluation (accuracy, F1-score, confusion matrix, ROC curves)

---

### 3. Final Model & Deployment Preparation

* Bayesian hyperparameter optimization (Optuna)
* Staged transfer learning
* Grad-CAM explainability
* Model preparation for deployment in the KneeAI system

---

## 🔬 Reproducibility

The notebooks are part of a fully reproducible pipeline that includes:

* Dataset structure and labels (`1_dataset/`)
* Model weights (`3_model/`)
* Hyperparameters and metadata (`4_metadata/`)
* Documentation (`5_documentation/`)

---

## ⚠️ Note

Users may need to update local dataset paths before executing the notebooks.

The GitHub repository contains the clinical deployment system, while Mendeley Data contains the full experimental workflow.
