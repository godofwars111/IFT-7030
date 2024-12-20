import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

def extract_wav2vec_features(file_path):
 
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.mean(dim=0).numpy()


    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(torch.tensor(waveform)).numpy()

   
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)

   
    embeddings = outputs.last_hidden_state
    if embeddings.size(1) > 300: 
        embeddings = embeddings[:, :300, :]
    elif embeddings.size(1) < 300:  
        pad_size = 300 - embeddings.size(1)
        embeddings = F.pad(embeddings, (0, 0, 0, pad_size), "constant", 0)

    return embeddings.squeeze(0)


class PN_Animaux_CNN(nn.Module):
    def __init__(self, input_size, num_classes, use_conv=True):
        super(PN_Animaux_CNN, self).__init__()
        self.use_conv = use_conv

        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        
        with torch.no_grad():
            sample_input = torch.randn(1, 1, 300, 768)  
            sample_output = self.pool(F.relu(self.conv1(sample_input)))
            sample_output = self.pool(F.relu(self.conv2(sample_output)))
            sample_output = self.pool(F.relu(self.conv3(sample_output)))
            flattened_size = sample_output.view(1, -1).size(1)  

       
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x1 = self.pool(F.relu(self.conv2(x)))
        x2 = self.pool(F.relu(self.conv3(x1)))

        x3 = x2.view(x.size(0), -1)  
        x4 = F.relu(self.fc1(x3))
        x5 = self.dropout(x4)
        x6 = self.fc2(x5)
        return x6




def compute_prototypes(embeddings, labels, num_classes):
    prototypes = []
    for cls in range(num_classes):
        class_embeddings = embeddings[labels == cls]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)



def episodic_data_sampler(data, labels, n_classes, n_support, n_query):
    unique_classes = random.sample(list(set(labels.tolist())), n_classes)

    support_set, query_set = [], []
    support_labels, query_labels = [], []

    for i, cls in enumerate(unique_classes):
        class_indices = np.where(labels == cls)[0]
        if len(class_indices) < n_support + n_query:
            sampled_indices = np.random.choice(class_indices, n_support + n_query, replace=True)
        else:
            sampled_indices = np.random.choice(class_indices, n_support + n_query, replace=False)

        support_set.extend(data[sampled_indices[:n_support]])
        query_set.extend(data[sampled_indices[n_support:]])
        support_labels.extend([i] * n_support)
        query_labels.extend([i] * n_query)

    support_set = torch.stack(support_set).unsqueeze(1)  
    query_set = torch.stack(query_set).unsqueeze(1)     


    support_set = support_set.squeeze(2)
    query_set = query_set.squeeze(2)

    support_labels = torch.tensor(support_labels)
    query_labels = torch.tensor(query_labels)

    return support_set, support_labels, query_set, query_labels


def entrain_prototypical_network(model, train_data, train_labels, optimizer, device, n_classes, n_support, n_query,
                               epochs):
    model.train()
    for epoch in range(epochs):
        support_set, support_labels, query_set, query_labels = episodic_data_sampler(
            train_data, train_labels, n_classes, n_support, n_query
        )
        support_set = support_set.unsqueeze(1)  
        query_set = query_set.unsqueeze(1)  


        support_set = support_set.squeeze(2)  
        query_set = query_set.squeeze(2)


        support_set, support_labels = support_set.to(device), support_labels.to(device)
        query_set, query_labels = query_set.to(device), query_labels.to(device)

        optimizer.zero_grad()

        support_embeddings = model(support_set)
        query_embeddings = model(query_set)

        prototypes = compute_prototypes(support_embeddings, support_labels, n_classes)
        distances = torch.cdist(query_embeddings, prototypes)
        log_p_y = F.log_softmax(-distances, dim=1)
        loss = F.nll_loss(log_p_y, query_labels)

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        
        
        
        
def test_prototypical_network_with_metrics(model, test_data, test_labels, device, n_classes, n_support, n_query):
    model.eval()
    all_q_labels = []
    all_pred = []

    with torch.no_grad():
        for _ in range(10):  
            support_set, support_labels, query_set, query_labels = episodic_data_sampler(
                test_data, test_labels, n_classes, n_support, n_query
            )

            support_set, support_labels = support_set.to(device), support_labels.to(device)
            query_set, query_labels = query_set.to(device), query_labels.to(device)

            support_embeddings = model(support_set)
            query_embeddings = model(query_set)

            prototypes = compute_prototypes(support_embeddings, support_labels, n_classes)
            distances = torch.cdist(query_embeddings, prototypes)
            _, predicted = torch.min(distances, dim=1)

            all_q_labels.extend(query_labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

   
    accuracy = accuracy_score(all_q_labels, all_pred)
    cm = confusion_matrix(all_q_labels, all_pred)
    report = classification_report(all_q_labels, all_pred, target_names=[str(i) for i in range(n_classes)])

    print(f"\nTest precision: {pre * 100:.2f}%")
    print("\nRapport Classificatio :")
    print(reporte)

 
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.title("Confusion Matrix")
    plt.xlabel("predite Labels")
    plt.ylabel("vrai Labels")
    plt.show()
    return accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size = 64
model = PN_Animaux_CNN(embedding_size,13).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


folder_sound = "../ani_sound/"
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

train_files, test_files, train_labels, test_labels = train_test_split(
    audio_files, encoded_labels, test_size=0.2, random_state=42
)

train_data = torch.stack([extract_wav2vec_features(fea) for fea in train_files])
test_data = torch.stack([extract_wav2vec_features(fea) for fea in test_files])
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

n_class = 13
n_support = 5
n_query = 5


entrain_prototypical_network(
    model, train_data, train_labels, optimizer, device, n_classes=n_class, n_support=n_support, n_query=n_query, epochs=20
)
precision = test_prototypical_network_with_metrics(
    model, test_data, test_labels, device, n_classes=n_class, n_support=n_support, n_query=n_query
)

print(f"Final Test precision: {precision:.2f}%")