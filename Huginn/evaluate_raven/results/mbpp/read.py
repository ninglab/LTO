import json
import numpy as np
import random

from typing import List
import numpy as np
import multiprocessing as mp
import traceback
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
    for idx, sample in enumerate(response_candidates):
        answer = sample["output"]
        count = counter_dict.get(answer, 0)
        counter_dict[answer] = count+1
        correctness_dict[answer] = sample["correctness"]
    key = max(counter_dict, key=counter_dict.get)
    value = correctness_dict[key]
    return key, value    
            
def conduct_weighted_majority_voting(response_candidates):
    counter_dict = {}
    correctness_dict = {}
    for idx, sample in enumerate(response_candidates):
        answer = sample["output"]
        count = counter_dict.get(answer, 0)
        counter_dict[answer] = count+sample["probability"]
        correctness_dict[answer] = sample["correctness"]
    key = max(counter_dict, key=counter_dict.get)
    value = correctness_dict[key]
    return key, value    

def evaluate_program(code_str, test_cases):
    try:
        local_env = {}
        exec(code_str, {}, local_env)
        for test in test_cases:
            exec(test, {}, local_env)
        return True, len(test_cases)  # All passed
    except Exception as e:
        passed = 0
        for test in test_cases:
            try:
                exec(code_str + "\n" + test, {}, {})
                passed += 1
            except:
                continue
        return False, passed

def run_tests(code, test_cases, return_dict):
    local_env = {}
    try:
        exec(code, {}, local_env)
        passed = 0
        for t in test_cases:
            try:
                exec(t, {}, local_env)
                passed += 1
            except:
                continue
        return_dict['passed'] = passed
    except Exception:
        return_dict['passed'] = 0

def evaluate_with_timeout(code, test_cases, timeout=2):
    manager = mp.Manager()
    return_dict = manager.dict()
    p = mp.Process(target=run_tests, args=(code, test_cases, return_dict))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return 0  # Timed out
    return return_dict.get('passed', 0)
     
correctness = []
correctness_samples = []
correctness_majority_voting = []
correctness_weighted_majority_voting = []
f = open("result_test.jsonl", "r")
for line in f.readlines():
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
        #sample["evaluation"] = evaluate_with_timeout(sample["output"], data["test_cases"])
        print("evaluating")
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
