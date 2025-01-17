import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
import joblib

# Feature Extraction Function
def extract_features(folder_path, label):
    data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            audio, sample_rate = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            data.append([mfccs_mean, label])
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    return data

# Load datasets
positive_sounds = extract_features("data/negative", 0)  # Non-scream
negative_sounds = extract_features("data/positive", 1)  # Scream

# Combine and prepare dataset
dataset = positive_sounds + negative_sounds
df = pd.DataFrame(dataset, columns=["features", "label"])
X = np.array(df["features"].tolist())
y = np.array(df["label"].tolist())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Model
print("Training SVM model...")
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Train MLP Model
print("Training MLP model...")
mlp_model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])
mlp_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
mlp_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
mlp_accuracy = mlp_model.evaluate(X_test, y_test, verbose=0)
print("MLP Accuracy:", mlp_accuracy[1])

# Save Models
print("Saving models...")
joblib.dump(svm_model, "svm_model.pkl")
mlp_model.save("mlp_model.h5")
print("Models saved successfully.")
