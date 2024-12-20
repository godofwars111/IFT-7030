
import os
from itertools import combinations
import torch
import torchaudio
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Animaux_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Animaux_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x1 = self.pool(F.relu(self.bn2(self.conv2(x))))
        x2 = F.relu(self.bn3(self.conv3(x1)))
        x3 = self.global_pool(x2).view(x2.size(0), -1) 
        x4 = self.dropout(x3)
        x5 = self.fc(x4)
        return x5


def prepar_audio(file_path, n_mels=128):
    waveforme, sample_rate = torchaudio.load(file_path)
    if waveforme.size(0) > 1: 
        waveforme = waveforme.mean(dim=0, keepdim=True)
    mel_trans = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    mel_spect = mel_trans(waveforme).squeeze(0)
    if mel_spect.size(1) < 256:
        padding = 256 - mel_spect.size(1)
        mel_spect = F.pad(mel_spect, (0, padding))
    else:
        mel_spect = mel_spect[:, :256]
    mel_spect = (mel_spect - mel_spect.mean()) / mel_spect.std()
    return mel_spect.unsqueeze(0)  


class Animaux_Dataset(Dataset):
    def __init__(self, fps, labels):
        self.file_paths = fps
        self.labels = labels

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        spectrogram = prepar_audio(self.fps[idx])
        label = self.labels[idx]
        return spectrogram, label


def entrain_and_evaluate(folder_sound, n_epochs=30, batch_size=32):


    for root, dirs, files in os.walk(folder_sound):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                label = os.path.basename(root)
                if label not in list_label:
                    list_label.append(label)

    n_comb = 13
    print(list_label)
    dict_combinations = list(combinations(list_label, n_comb))
    print(dict_combinations)
    index_comb = 0
    stop = False



    audio_files = []
    labels = []
    for root, dirs, files in os.walk(folder_sound):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                label = os.path.basename(root)
                audio_files.append(os.path.join(root, file))
                labels.append(label)



    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    X_train, X_test, y_train, y_test = train_test_split(
        audio_files, encoded_labels, test_size=0.2, random_state=42
    )

    train_dataset = Animaux_Dataset(X_train, y_train)
    test_dataset = Animaux_Dataset(X_test, y_test)


    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Animaux_CNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(n_epochs):
        model.train()
        for spect, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(spect.float())
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    all_q_labels = []
    all_pred = []

    with torch.no_grad():
        for spect, labels in test_loader:
            outputs = model(spect.float())
            _, predicted = torch.max(outputs, 1)
            all_q_labels.extend(labels.tolist())
            all_pred.extend(predicted.tolist())

    pre = accuracy_score(all_q_labels, all_predictions)
    cm = confusion_matrix(all_q_labels, all_pred)
    reporte = classification_report(all_query_labels, all_predictions, target_names=[str(i) for i in range(13)])

    print(f"\nTest precision: {pre * 100:.2f}%")
    print("\nRapport Classificatio :")
    print(reporte)

 
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.title("Confusion Matrix")
    plt.xlabel("predite Labels")
    plt.ylabel("vrai Labels")
    plt.show()






folder_sound ="../ani_sound/"


audio_files = []
labels = []
list_label = []





entrain_and_evaluate(folder_sound, num_epochs=30, batch_size=32)