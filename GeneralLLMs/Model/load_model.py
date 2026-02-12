import os
import sys
import time
import random

import argparse
import scipy.spatial
import math
import json
import torch
import torch.nn as nn

import numpy as np
import pickle
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#MODEL_REPO = "./Model/"
MODEL_REPO = ""


def load_base_model(args):
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "output_hidden_states": True
    }
    config = AutoConfig.from_pretrained(MODEL_REPO + args.model_name, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO + args.model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO + args.model_name, trust_remote_code=True)
    if "Qwen-7B-Chat" not in args.model_name:
        tokenizer.pad_token_id = 0

    return model, tokenizer, config