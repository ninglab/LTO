import json
import numpy as np
import random

from typing import List
import numpy as np
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
    
def conduct_majority_voting(response_candidates):
    counter_dict = {}
    correctness_dict = {}
    for sample in response_candidates:
        answer = sample["extracted_answer"][-1]
        count = counter_dict.get(answer, 0)
        counter_dict[answer] = count+1
        correctness_dict[answer] = sample["correctness"]
    key = max(counter_dict, key=counter_dict.get)
    value = correctness_dict[key]
    return key, value    
            
def conduct_weighted_majority_voting(response_candidates):
    counter_dict = {}
    correctness_dict = {}
    for sample in response_candidates:
        answer = sample["extracted_answer"][-1]
        count = counter_dict.get(answer, 0)
        counter_dict[answer] = count+sample["probability"]
        correctness_dict[answer] = sample["correctness"]
    key = max(counter_dict, key=counter_dict.get)
    value = correctness_dict[key]
    return key, value    
            
correctness = []
correctness_samples = []
correctness_majority_voting = []
correctness_weighted_majority_voting = []
f = open("result_test_new.jsonl", "r")
for line in list(f.readlines()):
    try:
        data = json.loads(line)
    except:
        continue
    idx = data["idx"]
    model_output = data["model_output"][:20]
    #correctness.append(random.choice(model_output)["correctness"])
    correctness.append(model_output[-1]["correctness"])
    max_probability = 0
    correctness_sample = False
    current_idx = 0
    for sample_idx, sample in enumerate(model_output):
        if sample["probability"] > max_probability:
            max_probability = sample["probability"]
            correctness_sample = sample["correctness"]
            current_idx = sample_idx
    correctness_sample = conduct_rejection_sampling(model_output,[sample["probability"] for sample in model_output], 1, 1e-2)
    correctness_samples.append(correctness_sample[0]["correctness"])
    correctness_majority_voting.append(conduct_majority_voting(model_output)[1])
    correctness_weighted_majority_voting.append(conduct_weighted_majority_voting(model_output)[1])
    #correctness_samples.append(correctness_sample)
    #print(correctness_sample)
f.close()
print("num_samples", len(correctness))
print("correct rate:", np.mean(np.array(correctness)))
print("correct rate majority voting:", np.mean(np.array(correctness_majority_voting)))
print("correct rate weighted majority voting:", np.mean(np.array(correctness_weighted_majority_voting)))
print("correct rate resampling:", np.mean(np.array(correctness_samples)))
