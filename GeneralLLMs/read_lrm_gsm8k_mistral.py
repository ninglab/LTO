import json
import numpy as np
import random

from typing import List
import numpy as np
import torch
from train_classifier_gsm8k_multiple_steps import TransformerClassifier
import os
import re
from pathlib import Path
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda, cpu)")
parser.add_argument("--seq_len", dest="seq_len", type=int, default=32, help="sequence length for evaluating the latent representations")
parser.add_argument("--embed_dim", dest="embed_dim", type=int, default=4096, help="embedding dimensionality")
parser.add_argument("--intermediate_size", dest="intermediate_size", type=int, default=14336, help="intermediate_size")
parser.add_argument("--num_head", dest="num_head", type=int, default=32, help="number of attention head")
parser.add_argument("--num_layer", dest="num_layer", type=int, default=2, help="number of attention layer")
parser.add_argument("--classifier_checkpoint_path", dest="classifier_checkpoint_path", type=str, default="./checkpoint_gsm8k_mistral.pth", help="checkpoint for the latent reward model")
parser.add_argument("--beta", dest="beta", type=float, default=0.001, help="beta to control the strength of KL constraint")
args = parser.parse_args()
classifier = TransformerClassifier(embed_dim=args.embed_dim, num_heads=args.num_head, ff_dim=args.intermediate_size, num_layers=args.num_layer, seq_len = args.seq_len).to(args.device)
classifier.load_state_dict(torch.load(args.classifier_checkpoint_path))
classifier.eval()
def conduct_rejection_sampling(response_candidates, response_rewards, num_samples, beta):

    candidates = {c: r for c, r in zip(range(len(response_candidates)), response_rewards)}
    accepted = []
    while len(accepted) < num_samples:
        max_reward = max(candidates.values())
        to_remove = []
        for c, r in candidates.items():
            u = np.random.uniform()
            if u >= np.exp((r - max_reward) / beta):
                continue
            accepted.append(c)
            to_remove.append(c)
        if len(accepted) == num_samples:
            break
        for c in to_remove:
            candidates.pop(c)
    return [response_candidates[idx] for idx in accepted]
    
def conduct_majority_voting(response_candidates, response_candidates_correctness):
    counter_dict = {}
    correctness_dict = {}
    for answer, correctness in zip(response_candidates, response_candidates_correctness):
        count = counter_dict.get(answer, 0)
        counter_dict[answer] = count+1
        correctness_dict[answer] = correctness
    key = max(counter_dict, key=counter_dict.get)
    value = correctness_dict[key]
    return key, value    
            
def conduct_weighted_majority_voting(response_candidates, response_candidates_correctness, response_candidates_probability):
    counter_dict = {}
    correctness_dict = {}
    for answer, correctness, probability in zip(response_candidates, response_candidates_correctness, response_candidates_probability):
        count = counter_dict.get(answer, 0)
        counter_dict[answer] = count+probability
        correctness_dict[answer] = correctness
    key = max(counter_dict, key=counter_dict.get)
    value = correctness_dict[key]
    return key, value    

def load_data(folder_path, embedding_path, groundtruth_answer_dict):

    embeddings = {}
    labels = {}
    output_numbers = {}
    print("loading data")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path)==False: continue
        f = open(file_path, "rb")
        data = pickle.load(f)
        try:
            idx = data["id"]
            correctness, number = check(data["output_seq"], groundtruth_answer_dict[idx])
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
        output_numbers[idx] = number
        f.close()
    return embeddings, labels, output_numbers



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
        return False, 99999
    try:
        if abs(float(number)-float(target))<0.01:
            return True, number
        else:
            return False, number
    except:
        return False, number
        
def load_groundtruth_answer(groundtruth_answer_file):
    groundtruth_answer_dict = {}
    f = open(groundtruth_answer_file, "r")
    for line in f.readlines():
        temp = json.loads(line)
        groundtruth_answer_dict[temp["id"]] = temp["answer"]
    f.close()
    return groundtruth_answer_dict        

file_path_list = [
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_1/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_2/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_3/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_4/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_5/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_6/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_7/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_8/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_9/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_10/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_11/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_12/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_13/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_14/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_15/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_16/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_17/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_18/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_19/",
"./results/OutputInfo/en/Output/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_20/",
]
    
embedding_path_list = [
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_1/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_2/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_3/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_4/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_5/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_6/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_7/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_8/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_9/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_10/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_11/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_12/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_13/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_14/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_15/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_16/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_17/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_18/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_19/",
"./results/OutputInfo/en/HiddenStates/mistralai/Mistral-7B-Instruct-v0.1/gsm8k_test_20/"
]
    
groundtruth_files_list = [
"./Data/gsm8k_test_1.jsonl",
"./Data/gsm8k_test_2.jsonl",
"./Data/gsm8k_test_3.jsonl",
"./Data/gsm8k_test_4.jsonl",
"./Data/gsm8k_test_5.jsonl",
"./Data/gsm8k_test_6.jsonl",
"./Data/gsm8k_test_7.jsonl",
"./Data/gsm8k_test_8.jsonl",
"./Data/gsm8k_test_9.jsonl",
"./Data/gsm8k_test_10.jsonl",
"./Data/gsm8k_test_11.jsonl",
"./Data/gsm8k_test_12.jsonl",
"./Data/gsm8k_test_13.jsonl",
"./Data/gsm8k_test_14.jsonl",
"./Data/gsm8k_test_15.jsonl",
"./Data/gsm8k_test_16.jsonl",
"./Data/gsm8k_test_17.jsonl",
"./Data/gsm8k_test_18.jsonl",
"./Data/gsm8k_test_19.jsonl",
"./Data/gsm8k_test_20.jsonl",
]



            
correctness = []
correctness_samples = []
correctness_majority_voting = []
correctness_weighted_majority_voting = []

embeddings_list = []
labels_list = []
output_numbers_list = []

for idx, files in enumerate(zip(file_path_list, embedding_path_list, groundtruth_files_list)):
    file_path, embedding_path, groundtruth_file = files
    groundtruth_answer_dict = load_groundtruth_answer(groundtruth_file)
    embeddings, labels, output_numbers = load_data(file_path, embedding_path, groundtruth_answer_dict)
    embeddings_list.append(embeddings)
    labels_list.append(labels)
    output_numbers_list.append(output_numbers)
    

num_samples = len(embeddings)
num_answers_per_sample = len(file_path_list)



for idx in range(num_samples):
    #correctness.append(random.choice(model_output)["correctness"])
    correctness.append(labels_list[-1][idx])
    max_probability = 0
    correctness_sample = False
    sample_probs = []
    model_output = []
    model_output_correctness = []
    for sample_idx in range(num_answers_per_sample):
        latent_representation = torch.tensor(embeddings_list[sample_idx][idx]).to(args.device)
        if latent_representation.dim() == 2:
            latent_representation = latent_representation.unsqueeze(0)
        with torch.no_grad():
            logits = classifier(latent_representation)
            probs = torch.sigmoid(logits).cpu().item()
            preds = probs > 0.5
        sample_probs.append(probs)
        model_output.append(output_numbers_list[sample_idx][idx])
        model_output_correctness.append(labels_list[sample_idx][idx])
        if probs > max_probability:
            max_probability = probs
            correctness_sample = probs
            current_idx = sample_idx
    correctness_sample = conduct_rejection_sampling(model_output_correctness, sample_probs, 1, args.beta)
    correctness_samples.append(correctness_sample[0])
    correctness_majority_voting.append(conduct_majority_voting(model_output, model_output_correctness)[1])
    correctness_weighted_majority_voting.append(conduct_weighted_majority_voting(model_output, model_output_correctness, sample_probs)[1])
    #correctness_samples.append(correctness_sample)
    #print(correctness_sample)
print("num_samples", len(correctness))
print("correct rate:", np.mean(np.array(correctness)))
print("correct rate majority voting:", np.mean(np.array(correctness_majority_voting)))
print("correct rate weighted majority voting:", np.mean(np.array(correctness_weighted_majority_voting)))
print("correct rate resampling:", np.mean(np.array(correctness_samples)))
