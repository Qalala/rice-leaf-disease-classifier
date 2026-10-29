# Rice Leaf Disease Classifier

An image classifier that detects rice leaf diseases by combining **deep feature extraction** with a **classical SVM**, exposed through a **Gradio** UI. A lighter classic OpenCV-feature + logistic-regression pipeline is also included for comparison.

Target classes:
- Bacterial Blight
- Blast
- Brown Spot
- Tungro

The stack is:
- **Primary pipeline**: ResNet50 (ImageNet features) → SVM → Gradio
- **Classic pipeline**: handcrafted OpenCV features → logistic regression

## Tech stack
- Python 3
- ResNet50 (TensorFlow / Keras feature extraction)
- SVM (scikit-learn)
- OpenCV
- Gradio
- NumPy / joblib
- Classic pipeline: OpenCV features + logistic regression

## Architecture (high level)

```
rice_leaf_disease.py                 [ResNet50 + SVM + Gradio]
├─ Load ResNet50 (include_top=False, pooling="avg")
├─ Walk class folders under "Rice Leaf Disease Images/"
├─ extract_features(image) → 2048-d vector
├─ LabelEncoder + train_test_split
├─ Train SVM (RBF)
├─ Save svm_model.pkl + label_encoder.pkl
└─ Launch Gradio predict UI

rice_leaf_classification.py          [classic baseline]
├─ Resize / grayscale
├─ statistical + histogram features
├─ LogisticRegression
├─ confusion_matrix.png
└─ prediction_results.csv

Rice Leaf Disease Images/
├─ Bacterialblight/
├─ Blast/
├─ Brownspot/
└─ Tungro/
```

### Primary pipeline details (`rice_leaf_disease.py`)
1. **ResNet50** is used only as a feature extractor (no custom classification head training).
2. Each leaf image is resized to `224×224`, color-converted, and passed through `preprocess_input`.
3. Features are flattened and collected with folder-name labels.
4. An **SVM (RBF)** is trained on the extracted features.
5. Metrics are printed (classification report + confusion matrix).
6. Model artifacts are saved with `joblib`.
7. **Gradio** launches a simple upload → prediction interface.

### Classic pipeline details (`rice_leaf_classification.py`)
- Faster / lighter dependency path
- Uses mean/std + histogram-style features rather than deep embeddings
- Good for quick experiments or machines where TensorFlow is heavy

### Dataset layout
This repository includes a **small sample** under `Rice Leaf Disease Images/` so the folder structure is clear. For serious training accuracy, replace/expand with the full dataset while keeping the same class folder names.

---

## Running the project

### Prerequisites
- Python 3.9+ recommended
- Sufficient RAM for ResNet50 feature extraction
- Optional GPU (not required; CPU works, just slower)

### Install dependencies

Primary (ResNet + Gradio) path:

```bash
pip install -r requirements.txt
```

Classic path only:

```bash
pip install -r requirements-classic.txt
```

### Run ResNet50 + SVM + Gradio

```bash
python rice_leaf_disease.py
```

What happens:
1. Features are extracted from `Rice Leaf Disease Images/`
2. SVM trains and evaluates
3. `svm_model.pkl` and `label_encoder.pkl` are written
4. Gradio opens in the browser for interactive predictions

> With only the tiny sample set, training is for pipeline verification, not production accuracy.

### Run classic pipeline

```bash
python rice_leaf_classification.py
```

Outputs include `confusion_matrix.png` and `prediction_results.csv`.

---

## How to navigate the repo
- `rice_leaf_disease.py` — CV-matching stack (ResNet50 + SVM + Gradio)
- `rice_leaf_classification.py` — classic baseline
- `requirements.txt` / `requirements-classic.txt` — dependency sets
- `Rice Leaf Disease Images/` — sample images / expected layout
- `confusion_matrix.png`, `prediction_results.csv` — example classic outputs

---

## Demo walkthrough
1. Install `requirements.txt`.
2. Confirm sample folders exist under `Rice Leaf Disease Images/`.
3. Run `python rice_leaf_disease.py`.
4. Wait for feature extraction + SVM training logs.
5. Open the Gradio URL printed in the terminal.
6. Upload a leaf image and read the predicted disease label.
7. (Optional) Swap in a fuller dataset and re-run for stronger results.

### Notes / limitations
- Sample images are intentionally small for GitHub size.
- First TensorFlow/ResNet run may download ImageNet weights.
- Gradio image color handling may differ slightly by Gradio version; if predictions look off, verify RGB/BGR conversion in the predict function for your environment.

---

## Project purpose
Classify rice leaf disease images using a hybrid deep-features + classical-classifier approach, with an interactive Gradio demo and a lighter classical baseline for comparison.
