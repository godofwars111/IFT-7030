import os
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from  sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to(device)
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")


def extract_wav2vec2_features(file_path):
    try:

        audio_input, _ = librosa.load(file_path, sr=16000)
        inputs = wav2vec2_processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            hidden_states = wav2vec2_model(inputs.input_values.to(device)).last_hidden_state
            features = torch.mean(hidden_states, dim=1).cpu().numpy()
        return features.squeeze()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


print(f"Debut :")
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


model_path = "svm_model_wav2vec2.joblib"
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