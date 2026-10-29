import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import joblib
import gradio as gr

# === 1. Load Pretrained ResNet50 ===
# We're only using it as a feature extractor (no classification layers)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# === 2. Set dataset path ===
dataset_path = "Rice Leaf Disease Images"  # CHANGE this to your dataset path

# === 3. Feature Extraction Function ===
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # ResNet50 input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = resnet_model.predict(img)
    return features.flatten()

# === 4. Load all images and extract features ===
features = []
labels = []

for label in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, label)
    if os.path.isdir(class_dir):
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            try:
                feat = extract_features(file_path)
                features.append(feat)
                labels.append(label)
            except:
                print(f"Skipping {file_path} due to an error.")

X = np.array(features)
le = LabelEncoder()
y = le.fit_transform(labels)

# === 5. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === 6. Train SVM Classifier ===
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train, y_train)

# === 7. Evaluate Model ===
y_pred = svm.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === 8. Save the Model and Encoder ===
joblib.dump(svm, "svm_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# === 9. GUI with Gradio ===
def predict_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = resnet_model.predict(img)
    prediction = svm.predict(features)
    label = le.inverse_transform(prediction)[0]
    return f"Predicted Disease: {label}"

# === 10. Launch Gradio Interface ===
iface = gr.Interface(fn=predict_image, inputs="image", outputs="text", title="Rice Leaf Disease Classifier")
iface.launch()
