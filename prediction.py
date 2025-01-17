import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model # type: ignore

# Load models
svm_model = joblib.load("svm_model.pkl")
mlp_model = load_model("mlp_model.h5")

def predict_audio(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)
    
    # First pass with SVM
    svm_pred = svm_model.predict(mfccs_mean)
    if svm_pred == 1:  # Potential scream
        mlp_pred = mlp_model.predict(mfccs_mean)
        return "Scream Detected" if mlp_pred[0][0] > 0.5 else "No Scream Detected"
    return "No Scream Detected"
