import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------
# Dataset
# -----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # (1, seq_len, d_model)
        return x + pos_emb

# -----------------------------
# Transformer Encoder Model
# -----------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, seq_len, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(embed_dim)
        #self.pos_encoder = LearnablePositionalEncoding(max_len=seq_len, d_model=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1)
        )

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        logits = self.classifier(x)
        return logits.squeeze(-1)

#TODO: develop padding funtions for varing lengths of latent embeddings

class ProcessRewardModel:
    def __init__(self, model):
        self.model = model
    
    def check(self, latent_representation):
        if latent_representation.dim() == 2:
            latent_representation = latent_representation.unqueeze(0)
        logits = model(latent_representation)
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        return {"prediction": preds, "probability": probs}
        


# -----------------------------
# Train and Evaluate
# -----------------------------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        #from torch.cuda import memory_summary
        #print(memory_summary(device=None, abbreviated=False))
    return total_loss / len(dataloader.dataset)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)

    return {"accuracy": "%.3f"%acc, "precision": "%.3f"%prec, "recall": "%.3f"%rec, "f1": "%.3f"%f1, "roc_auc": "%.3f"%roc_auc}


def load_data(args, file_path, embedding_path):
    embeddings = {}
    labels = {}
    f = open(file_path, "r")
    print("loading data")
    for line in f.readlines():
        #print(line)
        try:
            temp = json.loads(line.strip())
        
            idx = temp["idx"]
            correctness = int(temp["correctness"])
            embedding = np.squeeze(np.load(os.path.join(embedding_path,'{index}.npy'.format(index = idx))))[:args.seq_len+1]
            embeddings[idx] = embedding
            labels[idx] = correctness
        except:
            continue
    f.close()
    embeddings_list = []
    labels_list = []
    for i in range(len(embeddings)):
        try:
            embeddings_list.append(embeddings[i])
            labels_list.append(labels[i])
        except:
            continue
    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    return embeddings, labels
    
def load_data_token_localization(args, file_path, embedding_path, output_ids_path):
    embeddings = {}
    labels = {}
    output_ids = {}
    f = open(file_path, "r")
    print("loading data")
    for line in f.readlines():
        try:
            temp = json.loads(line.strip())
            idx = temp["idx"]
            correctness = int(temp["correctness"])
            embedding = np.squeeze(np.load(os.path.join(embedding_path,'{index}.npy'.format(index = idx))))
            new = np.reshape(embedding, (33, -1, 5280))
            #output_sequence = np.squeeze(np.load(os.path.join(output_ids_path,'{index}.npy'.format(index = idx))))
        except Exception as e:
            print("An error occurred:", e)
            continue
        #length = output_sequence.shape[-1]
        '''
        start = max(-50, -length)
        end = -1
        for i in range(-1, end, -1):
            if output_sequence[i] == 301:
                end = i
            if output_sequence[i] == 1319:
                start = i
                break
        if end<=start: end = -1
        '''
        start = -1
        end = 99999
        embeddings[idx] = np.mean(embedding[:,start+1:end,:], axis = 1)[:args.seq_len+1]
        labels[idx] = correctness
        #output_ids[idx] = output_sequence

    f.close()
    embeddings_list = []
    labels_list = []
    for i in range(len(embeddings)):
        try:
            embeddings_list.append(embeddings[i])
            labels_list.append(labels[i])
        except Exception as e:
            print("An error occurred during concatenating embeddings list:", e)
            pass
    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    return embeddings, labels
    
def load_data_from_multiple_files(args, file_path_list, embedding_path_list, output_ids_path_list):
    embeddings_list = []
    labels_list = []
    for idx, files in enumerate(zip(file_path_list, embedding_path_list, output_ids_path_list)):
        file_path, embedding_path, output_ids_path = files
        embeddings, labels = load_data_token_localization(args, file_path, embedding_path, output_ids_path)
        embeddings_list.append(embeddings)
        labels_list.append(labels)
    embeddings = np.concatenate(embeddings_list, axis = 0)
    labels = np.concatenate(labels_list, axis = 0)
    print("number of files loaded:", idx + 1)
    print("number of data samples", embeddings.shape[0])
    print(embeddings.shape)
    return embeddings, labels
    


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", dest="seq_len", type=int, default=32, help="sequence length for evaluating the latent representations")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="number of training epochs for the classifier")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--embed_dim", dest="embed_dim", type=int, default=5280, help="embedding dimensionality")
    parser.add_argument("--intermediate_size", dest="intermediate_size", type=int, default=17920, help="intermediate_size")
    parser.add_argument("--num_head", dest="num_head", type=int, default=55, help="number of attention head")
    parser.add_argument("--num_layer", dest="num_layer", type=int, default=2, help="number of attention layer")
    parser.add_argument("--classifier_file_save_path", dest="classifier_file_save_path", type=str, default=None, help="where to save the classifier checkpoint")
    args = parser.parse_args()
    print(args)
    #seq_len = 7
    file_path_list = [
    "./results/commonsenseqa/result_train1.jsonl",
    "./results/commonsenseqa/result_train2.jsonl",
    "./results/commonsenseqa/result_train3.jsonl",
    "./results/commonsenseqa/result_train4.jsonl",
    "./results/commonsenseqa/result_train5.jsonl"
    ]
    
    embedding_path_list = [
    "./results/commonsenseqa/train1/latent_embeddings/",
    "./results/commonsenseqa/train2/latent_embeddings/",
    "./results/commonsenseqa/train3/latent_embeddings/",
    "./results/commonsenseqa/train4/latent_embeddings/",
    "./results/commonsenseqa/train5/latent_embeddings/"
    ]
    
    output_ids_path_list = [
    "./results/commonsenseqa/train1/output_ids/",
    "./results/commonsenseqa/train2/output_ids/",
    "./results/commonsenseqa/train3/output_ids/",
    "./results/commonsenseqa/train4/output_ids/",
    "./results/commonsenseqa/train5/output_ids/",
    ]
    
    #embeddings_train, labels_train = load_data(args, "./result_training.jsonl", "/users/PCON0041/hwdu/recurrent-pretraining/evaluate_raven/latent_embeddings_training/")
    #embeddings_train, labels_train = load_data_token_localization(args, "./results/commonsenseqa/result_train.jsonl", "./results/commonsenseqa/train/latent_embeddings/", "./results/commonsenseqa/train/output_ids/")
    embeddings_train, labels_train = load_data_from_multiple_files(args, file_path_list, embedding_path_list, output_ids_path_list)
    #embeddings_test, labels_test = load_data_token_localization(args, "./results/commonsenseqa/result_test_latent_rep.jsonl", "./results/commonsenseqa/test_latent_rep/latent_embeddings/", "./results/commonsenseqa/test_latent_rep/output_ids/")
    embeddings_test, labels_test = load_data_from_multiple_files(args, file_path_list[:1], embedding_path_list[:1], output_ids_path_list[:1])
    dataset_train = EmbeddingDataset(embeddings_train, labels_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    
    dataset_test = EmbeddingDataset(embeddings_test, labels_test)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(embed_dim=args.embed_dim, num_heads=args.num_head, ff_dim=args.intermediate_size, num_layers=args.num_layer, seq_len = args.seq_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    criterion = nn.BCEWithLogitsLoss()
    
    best_metric = 0.0
    for epoch in range(args.epochs):
        print("start training")
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        metrics = evaluate_model(model, test_loader, device)
        test_metric = float(metrics["f1"])
        if args.classifier_file_save_path and test_metric > best_metric:
            torch.save(model.state_dict(), args.classifier_file_save_path) 
            best_metric = test_metric
            print("save checkpoint")
        print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")
        print("Metrics: ",  metrics)
