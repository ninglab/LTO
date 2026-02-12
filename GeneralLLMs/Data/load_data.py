import os
import jsonlines

dataset_path = "./Data/"

class DatasetInfo:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = []
        with open(dataset_path + self.dataset_name + ".jsonl", "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                self.data.append(item)
        self.data_size = len(self.data)


    def load_one_sample(self, idx):
        return self.data[idx]

