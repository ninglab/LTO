import os
import sys
import re
import pickle
import argparse

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import interpolate

project_root_path = os.environ["PROJECT_PATH"]
sys.path.append(project_root_path)
from Data.load_data import DatasetInfo
from config_pool import MODEL_POOL, DATASET_POOL, LANGUAGE_MAPPING
from match import AnswerParsing


class StandardEvaluation:
    def __init__(self, dataset_list):
        self.data_all = []
        self.data_size = 0
        for i, dataset in enumerate(dataset_list):
            data_loader = DatasetInfo(args.dataset)
            self.data_all.extend(data_loader.data)
            self.data_size += data_loader.data_size

    def std_eval(self, args):
        answerparsing = AnswerParsing(args.dataset)
        output_dir = os.path.join(project_root_path, f"OutputInfo/{args.language}/Output", args.model_name, args.dataset)
        coe_dir = os.path.join(project_root_path, f"OutputInfo/{args.language}/CoE", args.model_name, args.dataset)
        
        output_list, coe_list, binary_list = [], [], []
        acc = 0
        for i in range(self.data_size):
            sample = self.data_all[i]
            true_output = sample["answer"]

            with open(os.path.join(output_dir, f"{args.dataset}_{str(i)}.pkl"), 'rb') as file:
                output = pickle.load(file)
            pred_output = output["output_seq"]

            with open(os.path.join(coe_dir, f"{args.dataset}_{str(i)}.pkl"), 'rb') as file:
                coe = pickle.load(file)

            extracted_answer, binary = answerparsing.dataset_parse(pred_output, true_output, sample)
            if binary: acc += 1

            output_list.append(output)
            coe_list.append(coe)
            binary_list.append(binary)

        return round(acc / self.data_size, 3), output_list, coe_list, binary_list


class SelfEvaluation:
    def __init__(self, dataset_list):
        self.data_all = []
        self.data_size = 0
        for i, dataset in enumerate(dataset_list):
            data_loader = DatasetInfo(args.dataset)
            self.data_all.extend(data_loader.data)
            self.data_size += data_loader.data_size

    def self_eval(self, score_list, binary_list):
        fpr, tpr, thresholds = roc_curve(binary_list, score_list)
        auroc = auc(fpr, tpr)
        fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
        precision, recall, _ = precision_recall_curve(binary_list, score_list)
        aupr = auc(recall, precision)

        return round(auroc * 100, 2), round(fpr95 * 100, 2), round(aupr * 100, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--model_name", type=str, default="Llama-3-8B-Instruct", choices=MODEL_POOL)
    parser.add_argument("--dataset", type=str, default="mgsm", choices=DATASET_POOL)
    parser.add_argument("--language", type=str, default="en")

    args = parser.parse_args()

    stdeval = StandardEvaluation([args.dataset])
    acc, output_list, coe_list, binary_list = stdeval.std_eval(args)
    print(f"# Accuracy: {acc}")

    input_list = [output_list[i]["input_seq"] for i in range(len(output_list))]
    maxprob_list = [output_list[i]["maxprob"] for i in range(len(output_list))]
    ppl_list = [1 / output_list[i]["ppl"] for i in range(len(output_list))]
    entropy_list = [1 / output_list[i]["entropy"] for i in range(len(output_list))]
    coer_list = [coe_list[i]["R"] for i in range(len(coe_list))]
    coec_list = [coe_list[i]["C"] for i in range(len(coe_list))]

    selfeval = SelfEvaluation([args.dataset])
    maxprob_auroc, maxprob_fpr95, maxprob_aupr = selfeval.self_eval(maxprob_list, binary_list)
    ppl_auroc, ppl_fpr95, ppl_aupr = selfeval.self_eval(ppl_list, binary_list)
    entropy_auroc, entropy_fpr95, entropy_aupr = selfeval.self_eval(entropy_list, binary_list)
    coer_auroc, coer_fpr95, coer_aupr = selfeval.self_eval(coer_list, binary_list)
    coec_auroc, coec_fpr95, coec_aupr = selfeval.self_eval(coec_list, binary_list)

    print(f"{'maxprob_auroc'.rjust(13)}: {maxprob_auroc:.2f}    {'maxprob_fpr95'.rjust(13)}: {maxprob_fpr95:.2f}    {'maxprob_aupr'.rjust(13)}: {maxprob_aupr:.2f}")
    print(f"{'ppl_auroc'.rjust(13)}: {ppl_auroc:.2f}    {'ppl_fpr95'.rjust(13)}: {ppl_fpr95:.2f}    {'ppl_aupr'.rjust(13)}: {ppl_aupr:.2f}")
    print(f"{'entropy_auroc'.rjust(13)}: {entropy_auroc:.2f}    {'entropy_fpr95'.rjust(13)}: {entropy_fpr95:.2f}    {'entropy_aupr'.rjust(13)}: {entropy_aupr:.2f}")
    print(f"{'coer_auroc'.rjust(13)}: {coer_auroc:.2f}    {'coer_fpr95'.rjust(13)}: {coer_fpr95:.2f}    {'coer_aupr'.rjust(13)}: {coer_aupr:.2f}")
    print(f"{'coec_auroc'.rjust(13)}: {coec_auroc:.2f}    {'coec_fpr95'.rjust(13)}: {coec_fpr95:.2f}    {'coec_aupr'.rjust(13)}: {coec_aupr:.2f}")
