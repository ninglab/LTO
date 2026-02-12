import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import re
from pathlib import Path
import pickle
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
import itertools

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
        #print(self.embeddings[idx])
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

    def forward(self, x, padding_mask=None):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        #x = x.mean(dim=1)  # Global average pooling
        if padding_mask is not None:
            attention_mask = ~padding_mask  # True where real tokens
            attention_mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
            x = x * attention_mask  # zero out padded
            #print('attention_mask:')
            #print(attention_mask)
            sum_x = x.sum(dim=1)
            count = attention_mask.sum(dim=1).clamp(min=1)
            x = sum_x / count
        else:
            x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits.squeeze(-1)

def collate_fn(batch):
    embeddings, labels = zip(*batch)
    # embeddings is tuple of (seq_len_i, embed_dim) tensors

    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)  # (batch_size, max_seq_len, embed_dim)
    labels = torch.stack(labels)
    
    # Create padding mask: True for padded tokens
    padding_mask = (padded_embeddings.abs().sum(dim=-1) < 1e-5)  # sum over embed_dim; zero vectors are padded
    #print('padding_mask:')
    #print(padding_mask)
    return padded_embeddings, labels, padding_mask


# -----------------------------
# Train and Evaluate
# -----------------------------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y, batch_mask in dataloader:
        x, y, batch_mask = x.to(device), y.to(device), batch_mask.to(device)
        optimizer.zero_grad()
        logits = model(x, padding_mask=batch_mask)
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
        for x, y, batch_mask in dataloader:
            x, y, batch_mask = x.to(device), y.to(device), batch_mask.to(device)
            logits = model(x, padding_mask=batch_mask)
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

def check(answer, groundtruth):
    matches = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', answer)
    if len(matches)>0:
        matches = matches[-1]
    else:
        matches = ('', '')
    target = re.findall('#### (\\-?[0-9\\.\\,]+)', groundtruth)[-1]
    if matches[0]:
        number = matches[0]
    elif matches[1]:
        number = matches[1]
    else:
        return False
    try:
        if abs(float(number)-float(target))<0.01:
            return True
        else:
            return False
    except:
        return False

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
    #embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    return embeddings, labels
    
def load_data_token_localization(args, folder_path, embedding_path, groundtruth_answer_dict):

    embeddings = {}
    labels = {}
    answer_dict = {}
    print("loading data")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path)==False: continue
        f = open(file_path, "rb")
        data = pickle.load(f)
        try:
            idx = data["id"]
            correctness = check(data["output_seq"], groundtruth_answer_dict[idx])
            latent_thought_file = open(os.path.join(embedding_path, filename), "rb")
            embedding = pickle.load(latent_thought_file)["hidden_states"]
            embedding = np.array(embedding)
            #print(embedding.shape)
            latent_thought_file.close()
            
        except Exception as e:
            print("An error occurred:", e)
            continue


        #embeddings[idx] = np.mean(embedding[:,start+1:end,:], axis = 1)[:args.seq_len+1]
        embeddings[idx] = embedding
        labels[idx] = correctness
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
    #embeddings = np.array(embeddings_list)
    embeddings = embeddings_list
    print(embeddings_list[0].shape)
    labels = np.array(labels_list)
    return embeddings, labels
    
def load_data_from_multiple_files(args, file_path_list, embedding_path_list, groundtruth_files_list):
    embeddings_list = []
    labels_list = []
    for idx, files in enumerate(zip(file_path_list, embedding_path_list, groundtruth_files_list)):
        file_path, embedding_path, groundtruth_file = files
        groundtruth_answer_dict = load_groundtruth_answer(groundtruth_file)
        embeddings, labels = load_data_token_localization(args, file_path, embedding_path, groundtruth_answer_dict)
        #embeddings_list.append(embeddings)
        embeddings_list.extend(embeddings)
        labels_list.append(labels)
    #embeddings = np.concatenate(embeddings_list, axis = 0)
    #embeddings = list(itertools.chain(embeddings_list))
    labels = np.concatenate(labels_list, axis = 0)
    print("number of files loaded:", idx + 1)
    print("number of data samples", len(embeddings_list))
    #print(embeddings.shape)
    return embeddings_list, labels
    
def load_groundtruth_answer(groundtruth_answer_file):
    groundtruth_answer_dict = {}
    f = open(groundtruth_answer_file, "r")
    for line in f.readlines():
        temp = json.loads(line)
        groundtruth_answer_dict[temp["id"]] = temp["answer"]
    f.close()
    return groundtruth_answer_dict

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", dest="seq_len", type=int, default=40, help="sequence length for evaluating the latent representations")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="number of training epochs for the classifier")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--embed_dim", dest="embed_dim", type=int, default=5120, help="embedding dimensionality")
    parser.add_argument("--intermediate_size", dest="intermediate_size", type=int, default=13824, help="intermediate_size")
    parser.add_argument("--num_head", dest="num_head", type=int, default=40, help="number of attention head")
    parser.add_argument("--num_layer", dest="num_layer", type=int, default=2, help="number of attention layer")
    parser.add_argument("--classifier_file_save_path", dest="classifier_file_save_path", type=str, default=None, help="where to save the classifier checkpoint")
    args = parser.parse_args()
    print(args)
    #seq_len = 7
    file_path_list = [
    "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_1/",
    "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_2/",
    "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_3/",
    "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_4/",
    "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_5/",
    "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_6/",
    "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_7/",
    "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_8/",
    "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_9/",
    "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_10/",
    ]
    
    embedding_path_list = [
    "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_1/",
    "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_2/",
    "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_3/",
    "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_4/",
    "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_5/",
    "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_6/",
    "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_7/",
    "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_8/",
    "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_9/",
    "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_train_10/"

    ]
    
    groundtruth_files_list = [
    "./Data/gsm8k_train_1.jsonl",
    "./Data/gsm8k_train_2.jsonl",
    "./Data/gsm8k_train_3.jsonl",
    "./Data/gsm8k_train_4.jsonl",
    "./Data/gsm8k_train_5.jsonl",
    "./Data/gsm8k_train_6.jsonl",
    "./Data/gsm8k_train_7.jsonl",
    "./Data/gsm8k_train_8.jsonl",
    "./Data/gsm8k_train_9.jsonl",
    "./Data/gsm8k_train_10.jsonl"
    ]
    
    
    embeddings_train, labels_train = load_data_from_multiple_files(args, file_path_list, embedding_path_list, groundtruth_files_list)
    #embeddings_test, labels_test = load_data(args, "./result.jsonl", "/users/PCON0041/hwdu/recurrent-pretraining/evaluate_raven/latent_embeddings")
    embeddings_test, labels_test = load_data_token_localization(args, "./results/OutputInfo/en/Output/meta-llama/Llama-2-13b-chat-hf/gsm8k_test_1/", "./results/OutputInfo/en/HiddenStates/meta-llama/Llama-2-13b-chat-hf/gsm8k_test_1/", load_groundtruth_answer("./Data/gsm8k_test_1.jsonl"))
    dataset_train = EmbeddingDataset(embeddings_train, labels_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    dataset_test = EmbeddingDataset(embeddings_test, labels_test)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

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
        if test_metric > best_metric:
            best_metric = test_metric
        if args.classifier_file_save_path:
            torch.save(model.state_dict(), args.classifier_file_save_path) 
            print("save checkpoint")
        print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")
        print("Metrics: ",  metrics)
