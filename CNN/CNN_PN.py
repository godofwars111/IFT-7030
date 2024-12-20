import os
import torch
import torchaudio
import numpy as np
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
    def __init__(self, embedding_dim):
        super(Animaux_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x1 = self.pool(F.relu(self.bn2(self.conv2(x))))
        x2 = F.relu(self.bn3(self.conv3(x1)))
        x3 = self.global_pool(x2).view(x2.size(0), -1)
        x4 = self.fc(x3)
        return x4



class PN_Animaux_CNN(nn.Module):
    def __init__(self, embedding_dim):
        super(PrototypicalNetwork, self).__init__()
        self.embedding_net = Animaux_CNN(embedding_dim)

    def forward(self, x):
        if x.dim() == 5:  
            x = x.view(-1, x.size(2), x.size(3), x.size(4))  
        return self.embedding_net(x)

    def compute_prototypes(self, embeddings, labels, n_classes):
        prototypes = []
        for cls in range(n_classes):
            class_embeddings = embeddings[labels == cls]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def predict(self, embeddings, prototypes):
        distances = torch.cdist(embeddings, prototypes, p=2)
        return distances



def prepar_audio(file_path, n_mels=64):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    mel_spectrogram = mel_transform(waveform).squeeze(0)
    if mel_spectrogram.size(1) < 256:
        padding = 256 - mel_spectrogram.size(1)
        mel_spectrogram = F.pad(mel_spectrogram, (0, padding))
    else:
        mel_spectrogram = mel_spectrogram[:, :256]
    mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

    mel_spectrogram = mel_spectrogram.unsqueeze(0)  

    return mel_spectrogram

class Animaux_Dataset(Dataset):
    def __init__(self, file_paths, labels, n_way, k_shot, q_query):
        self.file_paths = file_paths
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        samples_per_class = self.k_shot + self.q_query
        selected_classes = np.random.choice(self.num_classes, self.n_way, replace=False)

        support_set, query_set, support_labels, query_labels = [], [], [], []
        class_mapping = {cls: i for i, cls in enumerate(selected_classes)}

        for cls in selected_classes:
            class_indices = np.where(self.encoded_labels == cls)[0]
            try:
                selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
            except Exception as e:
                selected_indices = np.random.choice(class_indices, samples_per_class, replace=True)


            support_indices = selected_indices[: self.k_shot]
            query_indices = selected_indices[self.k_shot :]

            support_set += [prepar_audio(self.file_paths[i]) for i in support_indices]
            query_set += [prepar_audio(self.file_paths[i]) for i in query_indices]
            support_labels += [class_mapping[cls]] * self.k_shot
            query_labels += [class_mapping[cls]] * self.q_query

  
        support_set = torch.stack(support_set) 
        query_set = torch.stack(query_set)     

    

        return support_set, torch.tensor(support_labels), query_set, torch.tensor(query_labels)




def entrain_prototypical_network(folder_sound, embedding_dim=128, num_epochs=30, batch_size=4, n_way=5, k_shot=5, q_query=15, test_size=0.2):
    global list_label
    list_label = []

 
    for root, dirs, files in os.walk(folder_sound):
        for dir_name in dirs:
            if dir_name not in list_label:
                list_label.append(dir_name)


    audio_files, labels = [], []
    for root, dirs, files in os.walk(folder_sound):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
                labels.append(os.path.basename(root))

   
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_files, labels, test_size=test_size, stratify=labels, random_state=42
    )

   
    train_dataset = Animaux_Dataset(train_files, train_labels, n_way, k_shot, q_query)
    test_dataset = Animaux_Dataset(test_files, test_labels, n_way, k_shot, q_query)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PN_Animaux_CNN(embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

     
        for support_set, support_labels, query_set, query_labels in train_loader:
            optimizer.zero_grad()

            support_set, query_set = support_set.float(), query_set.float()
            support_embeddings = model(support_set)  
            query_embeddings = model(query_set)


            support_labels = support_labels.view(-1) 


            prototypes = model.compute_prototypes(support_embeddings, support_labels, n_way)

            distances = model.predict(query_embeddings, prototypes)

        
            query_labels = query_labels.view(-1)  
            loss = F.cross_entropy(-distances, query_labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


   
    model.eval()
    with torch.no_grad():
        all_query_labels = []
        all_predictions = []
        for support_set, support_labels, query_set, query_labels in test_loader:
            support_set, query_set = support_set.float(), query_set.float()
            support_embeddings = model(support_set)
            query_embeddings = model(query_set)
            support_labels = support_labels.view(-1)
            query_labels = query_labels.view(-1)
            prototypes = model.compute_prototypes(support_embeddings, support_labels, n_way)
            distances = model.predict(query_embeddings, prototypes)
            predictions = torch.argmin(distances, dim=1)

            all_query_labels.extend(query_labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

        accuracy = accuracy_score(all_query_labels, all_predictions)
        cm = confusion_matrix(all_query_labels, all_predictions)
        report = classification_report(all_query_labels, all_predictions, target_names=[str(i) for i in range(13)])

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


# Folder path
folder_sound = "../ani_sound/"
results = entrain_prototypical_network(folder_sound, num_epochs=30, test_size=0.2)