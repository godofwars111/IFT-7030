import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pydub import AudioSegment
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import librosa

yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)


def extract_yamnet_features(file_path):
    try:
        processed_path = librosa.load(file_path, sr=16000)
        if not processed_path:
            return None


        audio_binary = tf.io.read_file(processed_path)
        waveform, sr = tf.audio.decode_wav(audio_binary)
        waveform = tf.squeeze(waveform, axis=-1) 


        scores, embeddings, spectrogram = yamnet_model(waveform)
        return tf.reduce_mean(embeddings, axis=0).numpy()  
    except Exception as e:
        print(f"Erreur  {e}")
        return None

print(f"Debut :")

folder_sound = '../ani_sound'

animals = [name for name in os.listdir(folder_sound) if os.path.isdir(os.path.join(folder_sound, name))]

data = []
feat = []


for label, animal in enumerate(animals):
    animal_folder = os.path.join(folder_sound, animal)
    if os.path.exists(animal_folder):
        for file_name in os.listdir(animal_folder):
            file_path = os.path.join(animal_folder, file_name)
            f_extract = extract_wav2vec2_features(file_path)
            if features is not None:
                data.append(f_extract)
                feat.append(label)

data = np.array(data)
feat = np.array(feat)


scaler = StandardScaler()
X = scaler.fit_transform(data)


X_train, X_test, y_train, y_test = train_test_split(data, feat, test_size=0.2, random_state=42)


svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)


model_path = "svm_model_yamnet.joblib"
joblib.dump(svm_model, model_path)
print(f"SVM model done")


y_pred = svm_model.predict(X_test)

print("Test Precision:", accuracy_score(y_test, y_pred))
print("\nRapport classt:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=animals, yticklabels=animals)
plt.title('Confusion Matrix')
plt.xlabel('Predit')
plt.ylabel('Realiter')
plt.show()


res = cross_val_score(svm_model, X, y, cv=5)  
print("\nCross-validation - Précision moyenne:", res.mean())
print("Scores fold ", res)

macro_precision = precision_score(y_test, y_pred, average='macro')
macro_recall = recall_score(y_test, y_pred, average='macro')
macro_f1 = f1_score(y_test, y_pred, average='macro')

micro_precision = precision_score(y_test, y_pred, average='micro')
micro_recall = recall_score(y_test, y_pred, average='micro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

print("\nMacro Precision:", macro_precision)
print("Macro Rappel:", macro_recall)
print("Macro F1 :", macro_f1)

print("\nMicro Precision:", micro_precision)
print("Micro Rappel:", micro_recall)
print("Micro F1 :", micro_f1)