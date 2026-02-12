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
        #self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.embeddings = embeddings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), self.labels[idx]


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
            embedding = np.squeeze(np.load(os.path.join(embedding_path,'{index}.npy'.format(index = idx)), mmap_mode='r'))
            new = np.reshape(embedding, (33, -1, 5280))
            #output_sequence = np.squeeze(np.load(os.path.join(output_ids_path,'{index}.npy'.format(index = idx)), mmap_mode='r'))
        except Exception as e:
            print("An error occurred:", e)
            continue
        if embedding.size==0:
            print("empty array")
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
    "./results/gsm8k/result_train1.jsonl",
    "./results/gsm8k/result_train2.jsonl",
    "./results/gsm8k/result_train3.jsonl",
    "./results/gsm8k/result_train4.jsonl",
    "./results/gsm8k/result_train5.jsonl",
    "./results/svamp/result_train1.jsonl",
    "./results/svamp/result_train2.jsonl",
    "./results/svamp/result_train3.jsonl",
    "./results/svamp/result_train4.jsonl",
    "./results/svamp/result_train5.jsonl",
    "./results/commonsenseqa/result_train1.jsonl",
    "./results/commonsenseqa/result_train2.jsonl",
    "./results/commonsenseqa/result_train3.jsonl",
    "./results/commonsenseqa/result_train4.jsonl",
    "./results/commonsenseqa/result_train5.jsonl",
    "./results/mbpp/result_train1.jsonl",
    "./results/mbpp/result_train2.jsonl",
    "./results/mbpp/result_train3.jsonl",
    "./results/mbpp/result_train4.jsonl",
    "./results/mbpp/result_train5.jsonl",
    "./results/mbpp/result_train6.jsonl",
    "./results/mbpp/result_train7.jsonl",
    "./results/mbpp/result_train8.jsonl",
    "./results/mbpp/result_train9.jsonl",
    "./results/mbpp/result_train10.jsonl",
    "./results/mbpp/result_train11.jsonl",
    "./results/mbpp/result_train12.jsonl",
    "./results/mbpp/result_train13.jsonl",
    "./results/mbpp/result_train14.jsonl",
    "./results/mbpp/result_train15.jsonl",
    "./results/mbpp/result_train16.jsonl",
    "./results/mbpp/result_train17.jsonl",
    "./results/mbpp/result_train18.jsonl",
    "./results/mbpp/result_train19.jsonl",
    "./results/mbpp/result_train20.jsonl",
    "./results/mbpp/result_train21.jsonl",
    "./results/mbpp/result_train22.jsonl",
    "./results/mbpp/result_train23.jsonl",
    "./results/mbpp/result_train24.jsonl",
    "./results/mbpp/result_train25.jsonl",
    "./results/mbpp/result_train26.jsonl",
    "./results/mbpp/result_train27.jsonl",
    "./results/mbpp/result_train28.jsonl",
    "./results/mbpp/result_train29.jsonl",
    "./results/mbpp/result_train30.jsonl",
    "./results/mbpp/result_train31.jsonl",
    "./results/mbpp/result_train32.jsonl",
    "./results/mbpp/result_train33.jsonl",
    "./results/mbpp/result_train34.jsonl",
    "./results/mbpp/result_train35.jsonl",
    "./results/mbpp/result_train36.jsonl",
    "./results/mbpp/result_train37.jsonl",
    "./results/mbpp/result_train38.jsonl",
    "./results/mbpp/result_train39.jsonl",
    "./results/mbpp/result_train40.jsonl",
    "./results/mbpp/result_train41.jsonl",
    "./results/mbpp/result_train42.jsonl",
    "./results/mbpp/result_train43.jsonl",
    "./results/mbpp/result_train44.jsonl",
    "./results/mbpp/result_train45.jsonl",
    "./results/mbpp/result_train46.jsonl",
    "./results/mbpp/result_train47.jsonl",
    "./results/mbpp/result_train48.jsonl",
    "./results/mbpp/result_train49.jsonl",
    "./results/mbpp/result_train50.jsonl",
    ]
    
    embedding_path_list = [
    "./results/gsm8k/train1/latent_embeddings/",
    "./results/gsm8k/train2/latent_embeddings/",
    "./results/gsm8k/train3/latent_embeddings/",
    "./results/gsm8k/train4/latent_embeddings/",
    "./results/gsm8k/train5/latent_embeddings/",
    "./results/svamp/train1/latent_embeddings/",
    "./results/svamp/train2/latent_embeddings/",
    "./results/svamp/train3/latent_embeddings/",
    "./results/svamp/train4/latent_embeddings/",
    "./results/svamp/train5/latent_embeddings/",
    "./results/commonsenseqa/train1/latent_embeddings/",
    "./results/commonsenseqa/train2/latent_embeddings/",
    "./results/commonsenseqa/train3/latent_embeddings/",
    "./results/commonsenseqa/train4/latent_embeddings/",
    "./results/commonsenseqa/train5/latent_embeddings/",
    "./results/mbpp/train1/latent_embeddings/",
    "./results/mbpp/train2/latent_embeddings/",
    "./results/mbpp/train3/latent_embeddings/",
    "./results/mbpp/train4/latent_embeddings/",
    "./results/mbpp/train5/latent_embeddings/",
    "./results/mbpp/train6/latent_embeddings/",
    "./results/mbpp/train7/latent_embeddings/",
    "./results/mbpp/train8/latent_embeddings/",
    "./results/mbpp/train9/latent_embeddings/",
    "./results/mbpp/train10/latent_embeddings/",
    "./results/mbpp/train11/latent_embeddings/",
    "./results/mbpp/train12/latent_embeddings/",
    "./results/mbpp/train13/latent_embeddings/",
    "./results/mbpp/train14/latent_embeddings/",
    "./results/mbpp/train15/latent_embeddings/",
    "./results/mbpp/train16/latent_embeddings/",
    "./results/mbpp/train17/latent_embeddings/",
    "./results/mbpp/train18/latent_embeddings/",
    "./results/mbpp/train19/latent_embeddings/",
    "./results/mbpp/train20/latent_embeddings/",
    "./results/mbpp/train21/latent_embeddings/",
    "./results/mbpp/train22/latent_embeddings/",
    "./results/mbpp/train23/latent_embeddings/",
    "./results/mbpp/train24/latent_embeddings/",
    "./results/mbpp/train25/latent_embeddings/",
    "./results/mbpp/train26/latent_embeddings/",
    "./results/mbpp/train27/latent_embeddings/",
    "./results/mbpp/train28/latent_embeddings/",
    "./results/mbpp/train29/latent_embeddings/",
    "./results/mbpp/train30/latent_embeddings/",
    "./results/mbpp/train31/latent_embeddings/",
    "./results/mbpp/train32/latent_embeddings/",
    "./results/mbpp/train33/latent_embeddings/",
    "./results/mbpp/train34/latent_embeddings/",
    "./results/mbpp/train35/latent_embeddings/",
    "./results/mbpp/train36/latent_embeddings/",
    "./results/mbpp/train37/latent_embeddings/",
    "./results/mbpp/train38/latent_embeddings/",
    "./results/mbpp/train39/latent_embeddings/",
    "./results/mbpp/train40/latent_embeddings/",
    "./results/mbpp/train41/latent_embeddings/",
    "./results/mbpp/train42/latent_embeddings/",
    "./results/mbpp/train43/latent_embeddings/",
    "./results/mbpp/train44/latent_embeddings/",
    "./results/mbpp/train45/latent_embeddings/",
    "./results/mbpp/train46/latent_embeddings/",
    "./results/mbpp/train47/latent_embeddings/",
    "./results/mbpp/train48/latent_embeddings/",
    "./results/mbpp/train49/latent_embeddings/",
    "./results/mbpp/train50/latent_embeddings/",
    ]
    
    output_ids_path_list = [
    "./results/gsm8k/train1/output_ids/",
    "./results/gsm8k/train2/output_ids/",
    "./results/gsm8k/train3/output_ids/",
    "./results/gsm8k/train4/output_ids/",
    "./results/gsm8k/train5/output_ids/",
    "./results/svamp/train1/output_ids/",
    "./results/svamp/train2/output_ids/",
    "./results/svamp/train3/output_ids/",
    "./results/svamp/train4/output_ids/",
    "./results/svamp/train5/output_ids/",
    "./results/commonsenseqa/train1/output_ids/",
    "./results/commonsenseqa/train2/output_ids/",
    "./results/commonsenseqa/train3/output_ids/",
    "./results/commonsenseqa/train4/output_ids/",
    "./results/commonsenseqa/train5/output_ids/",
    "./results/mbpp/train1/output_ids/",
    "./results/mbpp/train2/output_ids/",
    "./results/mbpp/train3/output_ids/",
    "./results/mbpp/train4/output_ids/",
    "./results/mbpp/train5/output_ids/",
    "./results/mbpp/train6/output_ids/",
    "./results/mbpp/train7/output_ids/",
    "./results/mbpp/train8/output_ids/",
    "./results/mbpp/train9/output_ids/",
    "./results/mbpp/train10/output_ids/",
    "./results/mbpp/train11/output_ids/",
    "./results/mbpp/train12/output_ids/",
    "./results/mbpp/train13/output_ids/",
    "./results/mbpp/train14/output_ids/",
    "./results/mbpp/train15/output_ids/",
    "./results/mbpp/train16/output_ids/",
    "./results/mbpp/train17/output_ids/",
    "./results/mbpp/train18/output_ids/",
    "./results/mbpp/train19/output_ids/",
    "./results/mbpp/train20/output_ids/",
    "./results/mbpp/train21/output_ids/",
    "./results/mbpp/train22/output_ids/",
    "./results/mbpp/train23/output_ids/",
    "./results/mbpp/train24/output_ids/",
    "./results/mbpp/train25/output_ids/",
    "./results/mbpp/train26/output_ids/",
    "./results/mbpp/train27/output_ids/",
    "./results/mbpp/train28/output_ids/",
    "./results/mbpp/train29/output_ids/",
    "./results/mbpp/train30/output_ids/",
    "./results/mbpp/train31/output_ids/",
    "./results/mbpp/train32/output_ids/",
    "./results/mbpp/train33/output_ids/",
    "./results/mbpp/train34/output_ids/",
    "./results/mbpp/train35/output_ids/",
    "./results/mbpp/train36/output_ids/",
    "./results/mbpp/train37/output_ids/",
    "./results/mbpp/train38/output_ids/",
    "./results/mbpp/train39/output_ids/",
    "./results/mbpp/train40/output_ids/",
    "./results/mbpp/train41/output_ids/",
    "./results/mbpp/train42/output_ids/",
    "./results/mbpp/train43/output_ids/",
    "./results/mbpp/train44/output_ids/",
    "./results/mbpp/train45/output_ids/",
    "./results/mbpp/train46/output_ids/",
    "./results/mbpp/train47/output_ids/",
    "./results/mbpp/train48/output_ids/",
    "./results/mbpp/train49/output_ids/",
    "./results/mbpp/train50/output_ids/",
    ]
    
    #embeddings_train, labels_train = load_data(args, "./result_training.jsonl", "/users/PCON0041/hwdu/recurrent-pretraining/evaluate_raven/latent_embeddings_training/")
    #embeddings_train, labels_train = load_data_token_localization(args, "./results/gsm8k/result_train.jsonl", "./results/gsm8k/train/latent_embeddings/", "./results/gsm8k/train/output_ids/")
    embeddings_train, labels_train = load_data_from_multiple_files(args, file_path_list, embedding_path_list, output_ids_path_list)
    #embeddings_test, labels_test = load_data(args, "./result.jsonl", "/users/PCON0041/hwdu/recurrent-pretraining/evaluate_raven/latent_embeddings")
    dataset_train = EmbeddingDataset(embeddings_train, labels_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    
    embeddings_test_gsm8k, labels_test_gsm8k = load_data_token_localization(args, "./results/gsm8k/result_train1.jsonl", "./results/gsm8k/train1/latent_embeddings/", "./results/gsm8k/train1/output_ids/")
    dataset_test_gsm8k = EmbeddingDataset(embeddings_test_gsm8k, labels_test_gsm8k)
    test_loader_gsm8k = DataLoader(dataset_test_gsm8k, batch_size=args.batch_size, shuffle=True)
    
    embeddings_test_svamp, labels_test_svamp = load_data_token_localization(args, "./results/svamp/result_train1.jsonl", "./results/svamp/train1/latent_embeddings/", "./results/svamp/train1/output_ids/")
    
    dataset_test_svamp = EmbeddingDataset(embeddings_test_svamp, labels_test_svamp)
    test_loader_svamp = DataLoader(dataset_test_svamp, batch_size=args.batch_size, shuffle=True)
    
    embeddings_test_commonsenseqa, labels_test_commonsenseqa = load_data_token_localization(args, "./results/commonsenseqa/result_train1.jsonl", "./results/commonsenseqa/train1/latent_embeddings/", "./results/commonsenseqa/train1/output_ids/")
    
    dataset_test_commonsenseqa = EmbeddingDataset(embeddings_test_commonsenseqa, labels_test_commonsenseqa)
    test_loader_commonsenseqa = DataLoader(dataset_test_commonsenseqa, batch_size=args.batch_size, shuffle=True)
    
    embeddings_test_mbpp, labels_test_mbpp = load_data_token_localization(args, "./results/mbpp/result_train1.jsonl", "./results/mbpp/train1/latent_embeddings/", "./results/mbpp/train1/output_ids/")
    
    dataset_test_mbpp = EmbeddingDataset(embeddings_test_mbpp, labels_test_mbpp)
    test_loader_mbpp = DataLoader(dataset_test_mbpp, batch_size=args.batch_size, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(embed_dim=args.embed_dim, num_heads=args.num_head, ff_dim=args.intermediate_size, num_layers=args.num_layer, seq_len = args.seq_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    
    best_metric = 0.0
    for epoch in range(args.epochs):
        print("start training")
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")
        test_metric = []
        metrics_gsm8k = evaluate_model(model, test_loader_gsm8k, device)
        print("Metrics GSM8K: ",  metrics_gsm8k)
        test_metric.append(float(metrics_gsm8k["f1"]))
        
        metrics_svamp = evaluate_model(model, test_loader_svamp, device)
        print("Metrics svamp: ",  metrics_svamp)
        test_metric.append(float(metrics_svamp["f1"]))
        
        metrics_commonsenseqa = evaluate_model(model, test_loader_commonsenseqa, device)
        print("Metrics commonsenseqa: ",  metrics_commonsenseqa)
        test_metric.append(float(metrics_commonsenseqa["f1"]))
        
        metrics_mbpp = evaluate_model(model, test_loader_mbpp, device)
        print("Metrics mbpp: ",  metrics_mbpp)
        test_metric.append(float(metrics_mbpp["f1"]))
        
        test_metric = np.mean(test_metric)
        
        if args.classifier_file_save_path and test_metric > best_metric:
            torch.save(model.state_dict(), args.classifier_file_save_path) 
            best_metric = test_metric
            print("save checkpoint")
        
        
