# Rice Leaf Disease Classifier

## Brief description
An image classifier that detects rice leaf diseases by combining **ResNet50** deep-feature extraction with a classical **SVM**, exposed through a **Gradio** UI. A lighter classic feature + logistic-regression script is also included.

## Project purpose
Classify rice leaf images into disease categories (Bacterial Blight, Blast, Brown Spot, Tungro) for an academic ML / computer-vision project.

## Project requirements
- Python 3.9+
- Install:

```bash
pip install -r requirements.txt
```

For the lighter classic pipeline only:

```bash
pip install -r requirements-classic.txt
```

### Dataset
This repo includes a **small sample** under `Rice Leaf Disease Images/` so the folder layout is clear.
For real training, download/use the full dataset and keep the same class folder names beside the scripts.

## How to navigate
- `rice_leaf_disease.py` — ResNet50 features + SVM + Gradio (matches the CV stack)
- `rice_leaf_classification.py` — classic OpenCV features + logistic regression
- `Rice Leaf Disease Images/` — sample images / expected layout
- `confusion_matrix.png`, `prediction_results.csv` — example outputs from the classic pipeline

## How to run

### ResNet50 + SVM + Gradio
```bash
python rice_leaf_disease.py
```
This trains on the images found in `Rice Leaf Disease Images/`, saves `svm_model.pkl` / `label_encoder.pkl`, then launches Gradio.

### Classic pipeline
```bash
python rice_leaf_classification.py
```
