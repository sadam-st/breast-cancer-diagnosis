# Breast Cancer Diagnosis

**Deep learning-based multiview multimodal feature fusion (MMFF) for breast cancer classification using mammograms and radiological features.**

## Features
- **Multiview Mammograms:** Includes four views: Cranio-Caudal (CC) and Medio-Lateral-Oblique (MLO) for both breasts.
- **Multimodal Data:** Combines imaging features from SE-ResNet50 with textual features extracted from radiological reports.
- **Late Feature Fusion:** Combines extracted features for better classification accuracy.
- **Evaluation Metrics:** Accuracy, AUC, F1-score, sensitivity, and precision.

## Dataset
- Includes mammograms and tabular data extracted from radiological reports.
- **Disclaimer:** The dataset is not included due to privacy concerns.


## Results
- **AUC (Benign vs Malignant):**
  - MMFF: `0.965`
  - Image-only: `0.545` (ResNet50)
  - Text-only: `0.842` (SVM)


## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-diagnosis.git
1. To train the model:
   ```bash
   python diagnosis_multimodal.py
