# Breast Cancer Diagnosis

**Deep learning-based multiview multimodal feature fusion (MMFF) for breast cancer classification using mammograms and radiological features.**

![Screenshot](images/diagram_se.png)

Fig: The proposed multiview multimodal architecture for breast cancer classification integrates imaging and textual data. ResNet50 with squeeze-and-excitation blocks extracts imaging features, while an ANN processes tabular metadata. Late fusion combines four mammographic views (LCC, LMLO, RCC, RMLO) with textual data (e.g., BIRADS scores), feeding an ANN classifier for accurate and robust breast cancer classification.

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
   git clone https://github.com/sadam-st/breast-cancer-diagnosis.git
1. To train the model:
   ```bash
   python diagnosis_multimodal.py
   
## How to Run
To cite our work:
  ```bash
@article{hussain2024multiview,
  title={Multiview Multimodal Feature Fusion for Breast Cancer Classification Using Deep Learning},
  author={Hussain, Sadam and Ali, Mansoor and Naseem, Usman and Avalos, Daly Betzabeth Avenda{\~n}o and Cardona-Huerta, Servando and Tamez-Pe{\~n}a, Jose Gerardo},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}

**## Contact**

Please feel free to contact me if you require any further information.
Email: sadamteewino@gmail.com


