import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
from tqdm import tqdm

# Define constants
DATA_DIR = "Rice Leaf Disease Images"
IMG_SIZE = 150  # Resize images to this size
RANDOM_STATE = 42

# Function to extract features from images
def extract_features(img_path, img_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract statistical features
    mean = np.mean(gray)
    std = np.std(gray)
    
    # Extract histogram features
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    
    # Extract GLCM texture features (Simplified)
    features = np.concatenate(([mean, std], hist))
    
    return features

# Load and preprocess data
def load_data():
    features = []
    labels = []
    classes = os.listdir(DATA_DIR)
    
    print("Loading and extracting features from images...")
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, class_name)
        print(f"Processing {class_name} images...")
        
        for img_file in tqdm(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, img_file)
            try:
                img_features = extract_features(img_path, IMG_SIZE)
                features.append(img_features)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(features), np.array(labels), classes

# Train logistic regression model
def train_model(X_train, y_train):
    print("Training logistic regression model...")
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train logistic regression with multinomial (softmax) option for multi-class
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Evaluate model
def evaluate_model(model, scaler, X_test, y_test, class_names):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Label the plot
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return accuracy, y_pred

# Main function
def main():
    # Load data
    features, labels, class_names = load_data()
    print(f"Loaded {len(features)} images with {len(class_names)} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Train model
    model, scaler = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, y_pred = evaluate_model(model, scaler, X_test, y_test, class_names)
    
    # Save model results
    results = pd.DataFrame({
        'Actual': [class_names[i] for i in y_test],
        'Predicted': [class_names[i] for i in y_pred]
    })
    results.to_csv('prediction_results.csv', index=False)
    
    print("\nModel training and evaluation completed successfully.")
    print(f"Model accuracy: {accuracy:.4f}")
    print("Confusion matrix has been saved as 'confusion_matrix.png'")
    print("Prediction results have been saved as 'prediction_results.csv'")

if __name__ == "__main__":
    main() 