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




def prepare_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_sample_width(2) 
        audio = audio.set_frame_rate(16000)  
        audio = audio.set_channels(1) 
        temp_path = file_path.replace(".wav", "_processed.wav")
        audio.export(temp_path, format="wav")
        return temp_path
    except Exception as e:
        print(f"Erreur lors de la conversion du fichier {file_path}: {e}")
        return None

print(f"Debut :")

folder_sound = '../ani_sound'

animals = [name for name in os.listdir(folder_sound) if os.path.isdir(os.path.join(folder_sound, name))]

data = []
label = []


for label, animal in enumerate(animals):
    animal_folder = os.path.join(folder_sound, animal)
    if os.path.exists(animal_folder):
        for file_name in os.listdir(animal_folder):
            file_path = os.path.join(animal_folder, file_name)
            f_extract = prepare_audio(file_path)
            data.append(f_extract)
            label.append(label)

data = np.array(data)
label = np.array(label)


scaler = StandardScaler()
X = scaler.fit_transform(data)


X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)


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


res = cross_val_score(svm_model, X, y, cv=5)  # 5-fold cross-validation
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