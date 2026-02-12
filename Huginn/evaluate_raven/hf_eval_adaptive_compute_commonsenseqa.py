from pathlib import Path
from typing import Literal, Optional, Union
import os
from pathlib import Path
import re
import sys
import json
import numpy as np

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer, GenerationConfig
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import stop_sequences_criteria

from recpre.raven_modeling_minimal import CausalSelfAttention, RavenForCausalLM
from evaluate_raven.quick_checkpoint_eval import prepare_results

from train_classifier import TransformerClassifier


system_prompt = """You are Huginn, an AI assistant who embodies careful thought and deliberation. Your responses demonstrate:

Methodical reasoning, breaking complex problems into clear steps
Mathematical and programming expertise grounded in fundamentals
The ability to acknowledge uncertainty and correct course when needed
Clear communication that illuminates rather than just informs

When engaging with questions, you first seek to understand their deeper structure before answering. Like your namesake who flew the nine worlds seeking wisdom, you explore problems from multiple angles, helping users build genuine understanding rather than providing shallow answers.
You express warmth and intellectual curiosity while maintaining professionalism. When faced with errors or confusion, you model honest reflection and careful correction. Your goal is not just to provide answers, but to help humans develop clearer, deeper thinking."""
from dataclasses import dataclass
@dataclass
class Message:
    role: str
    content: str

def update_huggingface_implementation(model):
    """This function selectively updates function implementations in the huggingface model."""
    import types
    # for name, module in model.named_modules():
    #     if module.__class__.__name__ == "CausalSelfAttention":
    #         module.forward = types.MethodType(CausalSelfAttention.forward, module)
    model.generate = types.MethodType(RavenForCausalLM.generate, model)
    model.generate_with_adaptive_compute = types.MethodType(RavenForCausalLM.generate_with_adaptive_compute, model)
    model.forward = types.MethodType(RavenForCausalLM.forward, model)


class HuginnWrapper(HFLM):
    """Wrapper for Huginn model using lm_eval, extending HFLM."""

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        *,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        criterion: Optional[Literal["entropy-diff", "latent-diff", "minp-kl", "argmax-stability"]] = "entropy-diff",
        exit_threshold: Optional[Union[str, float, int]] = "auto",
        lookup_strategy: str = "full",
        continuous_compute: bool = False,
        latent_dampening: bool = False,
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained,
            backend,
            revision,
            subfolder,
            tokenizer,
            truncation,
            logits_cache,
            max_length,
            device,
            dtype,
            batch_size,
            max_batch_size,
            trust_remote_code,
            use_fast_tokenizer,
            add_bos_token,
            prefix_token_id,
            parallelize,
            max_memory_per_gpu,
            max_cpu_memory,
            offload_folder,
            peft,
            delta,
            autogptq,
            gptqmodel,
            **kwargs
        )
        self.criterion = criterion
        self.exit_threshold = exit_threshold
        self.lookup_strategy = lookup_strategy
        self.continuous_compute = continuous_compute
        self.latent_dampening = latent_dampening
        update_huggingface_implementation(self.model)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # The generation configs is only used by the custom generate function call, 
        # whereas the standard generate function call uses these args directly passed in.
        # So we need to pass both, and have the dispatching generate function call decide which to use.
        generation_config = GenerationConfig(
            max_length=max_length,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        print("context:", context)
        print("max_length:", max_length)
        print("stop:", stop)
        print("generation_kwargs:", generation_kwargs)
        args_dict = {}
        result = super()._model_generate(
            context,
            max_length,
            stop,
            generation_config=generation_config,
            criterion=self.criterion, 
            exit_threshold=self.exit_threshold, 
            cache_kwargs={"lookup_strategy": self.lookup_strategy},
            continuous_compute=self.continuous_compute,
            latent_dampening=self.latent_dampening,
            my_args = args_dict,
            **generation_kwargs
        )
        print("new_args:", args_dict)
        return result


def evaluate_single_task(
    task_name="commonsenseqa",
    model_name="tomg-group-umd/huginn-0125",
    device="cuda",
    batch_size=16,
    num_fewshot=5,
    limit=None,
    criterion: Optional[Literal["entropy-diff", "latent-diff", "minp-kl", "argmax-stability"]] = "entropy-diff",
    exit_threshold: Optional[Union[str, float, int]] = "auto",
    num_steps=32,
    lookup_strategy="full",
    continuous_compute=False,
    latent_dampening=False,
):
    config_args = {
        "task_name": task_name,
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "criterion": criterion,
        "exit_threshold": exit_threshold,
        "num_steps": num_steps,
        "lookup_strategy": lookup_strategy,
    }

    print(f"Evaluating {model_name} on {task_name} with config: {config_args}")
    model = HuginnWrapper(
        pretrained=model_name,
        device=device,
        batch_size=batch_size,
        trust_remote_code=True,
        dtype="bfloat16",
        criterion=criterion,
        exit_threshold=exit_threshold,
        lookup_strategy=lookup_strategy,
        continuous_compute=continuous_compute,
        latent_dampening=latent_dampening,
    )
    results = evaluator.simple_evaluate(
        model=model,
        tasks=[task_name],
        num_fewshot=num_fewshot,
        limit=limit,
        gen_kwargs=f"num_steps={num_steps}",
    )
    if results is not None:
        results["config_args"] = config_args
        prepare_results(results, Path(f"{task_name}_results.json"))
    return results
    
def evaluate_with_system_prompt(args):
    stage = args.stage
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=False, # set to True if recpre lib not loaded
                                             torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    update_huggingface_implementation(model)
    config = GenerationConfig(max_length = 4096, max_new_tokens = 512, stop_strings=["<|end_text|>", "<|end_turn|>", "Question:", "Q:"], 
                          do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None, 
                          return_dict_in_generate=False,
                          eos_token_id=65505,bos_token_id=65504,pad_token_id=65509)
    f = open(args.dataset_path, "r")
    data = []
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    mkdir("./results/commonsenseqa/{stage}/".format(stage = stage))
    mkdir("./results/commonsenseqa/{stage}/latent_embeddings/".format(stage = stage))
    mkdir("./results/commonsenseqa/{stage}/output_ids/".format(stage = stage))
    for idx, sample in enumerate(data):
        if Path('./results/commonsenseqa/{stage}/output_ids/{index}.npy'.format(stage = stage, index = idx)).exists():
            continue
        messages = []
        messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? \nA. ignore \nB. enforce \nC. authoritarian \nD. yell at \nE. avoid"))
        
        messages.append(Message(role="assistant", content="The sentence is: The sanctions against the school were a punishing blow, and they seemed to ___ the efforts the school had made to change. We are looking for a word that fits the meaning: the sanctions were a punishment despite the school’s efforts to improve. So the sanctions seemed to disregard or ignore the positive steps the school had taken. Therefore, choice A ignore is the best fit.\n#### A"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: Sammy wanted to go to where the people were.  Where might he go? \nA. race track \nB. populated areas \nC. the desert \nD. apartment \nE. roadblock"))
        
        messages.append(Message(role="assistant", content="Sammy wants to go where the people are. Among the options, choice B populated areas directly refers to places with many people, making it the most suitable choice. The other options may or may not have people, but they are not as populated.\n#### B"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nGoogle Maps and other highway and street GPS services have replaced what? \nA. united states \nB. mexico \nC. countryside \nD. atlas \nE. oceans"))
        
        
        messages.append(Message(role="assistant", content="Google Maps and other GPS navigation services have largely replaced the use of physical maps and atlases for finding routes, directions, and locations. An atlas is a collection of maps, often in book form, used before digital navigation tools became common. Other options are places or types of locations, not tools or systems that GPS services replace. Therefore, choice D atlas is the best fit.\n#### D"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: The fox walked from the city into the forest, what was it looking for? \nA. pretty flowers. \nB. hen house \nC. natural habitat \nD. storybook \nE. dense forest"))
        
        messages.append(Message(role="assistant", content="The fox walked from the city into the forest, which suggests it was moving from an urban environment back to where it naturally belongs. A fox’s natural habitat is the forest, where it can find shelter, food, and safety.\n#### C\n"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: What home entertainment equipment requires cable? \nA. radio shack \nB. substation \nC. cabinet \nD. television \nE. desk"))
        
        messages.append(Message(role="assistant", content="Televisions often require cable connections for receiving cable TV signals or connecting to cable boxes and other media sources. The other options are not types of home entertainment equipment that require cable connections.\n#### D"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: The only baggage the woman checked was a drawstring bag, where was she heading with it? \nA. garbage can \nB. military \nC. jewelry store \nD. safe \nE. airport"))
        
        messages.append(Message(role="assistant", content="Because checking baggage is something done at airports before a flight, and a drawstring bag could be a lightweight personal bag she checked in. The other options don't typically involve checking baggage.\n#### E"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: What do people use to absorb extra ink from a fountain pen? \nA. shirt pocket \nB. calligrapher's hand \nC. inkwell \nD. desk drawer \nE. blotter"))
        
        messages.append(Message(role="assistant", content="A blotter is specially designed paper used to absorb excess ink from a fountain pen to prevent smudging.\n#### E"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: Where is a business restaurant likely to be located? \nA. business sector \nB. at hotel \nC. mall \nD. town \nE. yellow pages"))
        
        messages.append(Message(role="assistant", content="A business restaurant is typically intended to serve professionals, often for business lunches, meetings, or formal dining. Such restaurants are most likely to be located in a business sector, where many offices, corporate buildings, and professionals are concentrated.\n#### A"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: {question}".format(question = sample["question"])))
        
        formatted_messages = [
        {"role": "Huginn" if m.role == "assistant" else m.role, "content": m.content.strip()} for m in messages
        ]
        chat_input = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        #print(chat_input)
        input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(args.device)
        my_args = {}
        output = model.generate_with_adaptive_compute(input_ids, generation_config = config, criterion = args.criterion, exit_threshold = args.exit_threshold, lookup_strategy = args.lookup_strategy,  device = args.device, num_steps = args.num_steps, dtype="bfloat16", my_args = my_args)
        #print(output)
        decoded_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    

        target = sample["answer"]
        pattern = r"\s*(?:choice|Choice|####\s*)?([A-E])"
        try:
            matches = re.findall(pattern, decoded_text, flags=re.MULTILINE)
        except:
            matches = None
        
        if matches:
            correctness = matches[-1]==target
        else:
            correctness = False
        print(decoded_text, matches, target)
        output = decoded_text
        latent_embeddings = my_args.pop("latent_embeddings")
        np.save('./results/commonsenseqa/{stage}/latent_embeddings/{index}.npy'.format(stage = stage, index = idx), latent_embeddings)
        #np.save('/users/PCON0041/hwdu/recurrent-pretraining/evaluate_raven/latent_embeddings_training/{index}.npy'.format(index = idx), latent_embeddings)
        output_ids = my_args.pop("output_ids")
        np.save('./results/commonsenseqa/{stage}/output_ids/{index}.npy'.format(stage = stage, index = idx), output_ids)
        temp = {"idx": idx, "question": sample["question"], "answer": sample["answer"], "model_output": output, "extracted_answer": matches,"latent_embedding_evaluation_scores": my_args, "correctness": correctness}
        file_to_save = open("./results/commonsenseqa/result_{stage}.jsonl".format(stage = stage), "a")
        file_to_save.write(json.dumps(temp)+"\n")
        print(temp)
        file_to_save.close()

def evaluate_with_resampling(args):
    stage = args.stage
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=False, # set to True if recpre lib not loaded
                                             torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device)
    model.eval()
    

    
    classifier = TransformerClassifier(embed_dim=args.embed_dim, num_heads=args.num_head, ff_dim=args.intermediate_size, num_layers=args.num_layer, seq_len = args.seq_len).to(args.device)
    classifier.load_state_dict(torch.load('checkpoint_commonsenseqa.pth'))
    #classifier = classifier.half()
    classifier.eval()
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    update_huggingface_implementation(model)
    config = GenerationConfig(max_length = 4096, max_new_tokens = 512, stop_strings=["<|end_text|>", "<|end_turn|>", "Question:", "Q:"], 
                          do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None, 
                          return_dict_in_generate=False,
                          eos_token_id=65505,bos_token_id=65504,pad_token_id=65509)
    f = open(args.dataset_path, "r")
    data = []
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    mkdir("./results/commonsenseqa/{stage}/".format(stage = stage))
    mkdir("./results/commonsenseqa/{stage}/latent_embeddings/".format(stage = stage))
    mkdir("./results/commonsenseqa/{stage}/output_ids/".format(stage = stage))
    for idx, sample in enumerate(data):
        if Path('./results/commonsenseqa/{stage}/output_ids/{index}'.format(stage = stage, index = idx)).exists():
            continue
        messages = []
        messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? \nA. ignore \nB. enforce \nC. authoritarian \nD. yell at \nE. avoid"))
        
        messages.append(Message(role="assistant", content="The sentence is: The sanctions against the school were a punishing blow, and they seemed to ___ the efforts the school had made to change. We are looking for a word that fits the meaning: the sanctions were a punishment despite the school’s efforts to improve. So the sanctions seemed to disregard or ignore the positive steps the school had taken. Therefore, choice A ignore is the best fit.\n#### A"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: Sammy wanted to go to where the people were.  Where might he go? \nA. race track \nB. populated areas \nC. the desert \nD. apartment \nE. roadblock"))
        
        messages.append(Message(role="assistant", content="Sammy wants to go where the people are. Among the options, choice B populated areas directly refers to places with many people, making it the most suitable choice. The other options may or may not have people, but they are not as populated.\n#### B"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nGoogle Maps and other highway and street GPS services have replaced what? \nA. united states \nB. mexico \nC. countryside \nD. atlas \nE. oceans"))
        
        
        messages.append(Message(role="assistant", content="Google Maps and other GPS navigation services have largely replaced the use of physical maps and atlases for finding routes, directions, and locations. An atlas is a collection of maps, often in book form, used before digital navigation tools became common. Other options are places or types of locations, not tools or systems that GPS services replace. Therefore, choice D atlas is the best fit.\n#### D"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: The fox walked from the city into the forest, what was it looking for? \nA. pretty flowers. \nB. hen house \nC. natural habitat \nD. storybook \nE. dense forest"))
        
        messages.append(Message(role="assistant", content="The fox walked from the city into the forest, which suggests it was moving from an urban environment back to where it naturally belongs. A fox’s natural habitat is the forest, where it can find shelter, food, and safety.\n#### C\n"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: What home entertainment equipment requires cable? \nA. radio shack \nB. substation \nC. cabinet \nD. television \nE. desk"))
        
        messages.append(Message(role="assistant", content="Televisions often require cable connections for receiving cable TV signals or connecting to cable boxes and other media sources. The other options are not types of home entertainment equipment that require cable connections.\n#### D"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: The only baggage the woman checked was a drawstring bag, where was she heading with it? \nA. garbage can \nB. military \nC. jewelry store \nD. safe \nE. airport"))
        
        messages.append(Message(role="assistant", content="Because checking baggage is something done at airports before a flight, and a drawstring bag could be a lightweight personal bag she checked in. The other options don't typically involve checking baggage.\n#### E"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: What do people use to absorb extra ink from a fountain pen? \nA. shirt pocket \nB. calligrapher's hand \nC. inkwell \nD. desk drawer \nE. blotter"))
        
        messages.append(Message(role="assistant", content="A blotter is specially designed paper used to absorb excess ink from a fountain pen to prevent smudging.\n#### E"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: Where is a business restaurant likely to be located? \nA. business sector \nB. at hotel \nC. mall \nD. town \nE. yellow pages"))
        
        messages.append(Message(role="assistant", content="A business restaurant is typically intended to serve professionals, often for business lunches, meetings, or formal dining. Such restaurants are most likely to be located in a business sector, where many offices, corporate buildings, and professionals are concentrated.\n#### A"))
        
        messages.append(Message(role="user", content="Provide your answer to the multi-choice questions in the format: <explanation> #### <A/B/C/D/E>\n\nQuestion: {question}".format(question = sample["question"])))
        
        formatted_messages = [
        {"role": "Huginn" if m.role == "assistant" else m.role, "content": m.content.strip()} for m in messages
        ]
        chat_input = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        #print(chat_input)
        input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(args.device)
        output_list = []
        mkdir("./results/commonsenseqa/{stage}/latent_embeddings/{index}".format(stage = stage, index = idx))
        mkdir("./results/commonsenseqa/{stage}/output_ids/{index}".format(stage = stage, index = idx))
        for i in range(args.num_samples):
            my_args = {}
            output = model.generate_with_adaptive_compute(input_ids, generation_config = config, criterion = args.criterion, exit_threshold = args.exit_threshold, lookup_strategy = args.lookup_strategy,  device = args.device, num_steps = args.num_steps, dtype="bfloat16", my_args = my_args)
        #print(output)
            decoded_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            
            target = sample["answer"]
            pattern = r"\s*(?:choice|Choice|####\s*)?([A-E])"
        
            try:
                matches = re.findall(pattern, decoded_text, flags=re.MULTILINE)
            except:
                matches = None
        
            if matches:
                correctness = matches[-1]==target
            else:
                correctness = False
            output = decoded_text
            
            temp = torch.tensor(my_args.pop("latent_embeddings"), dtype=torch.float32).mean(dim = 1)
            latent_representation = temp.to(args.device)
            np.save('./results/commonsenseqa/{stage}/latent_embeddings/{index}/{index2}.npy'.format(stage = stage, index = idx, index2 = i), temp.numpy())
            output_ids = my_args.pop("output_ids")
            np.save('./results/commonsenseqa/{stage}/output_ids/{index}/{index2}.npy'.format(stage = stage, index = idx, index2 = i), output_ids)
            if latent_representation.dim() == 2:
                latent_representation = latent_representation.unsqueeze(0)
            with torch.no_grad():
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
                logits = classifier(latent_representation)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)
                probs = torch.sigmoid(logits).cpu().item()
                preds = probs > 0.5
            output_list.append({"output": output, "extracted_answer": matches, "correctness": correctness, "prediction": preds, "probability": probs})
            print(output_list[-1])
            #if probs > args.correctness_threshold:
            #    break
            
        temp = {"idx": idx, "question": sample["question"], "answer": sample["answer"], "model_output": output_list}
        file_to_save = open("./results/commonsenseqa/result_{stage}.jsonl".format(stage = stage), "a")
        file_to_save.write(json.dumps(temp)+"\n")
        file_to_save.close()

def mkdir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a model on a task with adaptive compute.")
    parser.add_argument("--task-name", dest="task_name", type=str, default="commonsenseqa", help="Task to evaluate on")
    parser.add_argument("--model-name", dest="model_name", type=str, default="tomg-group-umd/huginn-0125", help="Model to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda, cpu)")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default="./commonsenseqa/test.jsonl", help="where the dataset is located")
    parser.add_argument("--stage", dest="stage", type=str, default="train", help="stage identifier")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num-fewshot", dest="num_fewshot", type=int, default=8, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to evaluate")
    parser.add_argument("--criterion", type=str, default="entropy-diff", 
                        choices=["entropy-diff", "latent-diff", "minp-kl", "argmax-stability", "none"],
                        help="Criterion for adaptive compute. Pass `none` to disable adaptive compute.")
    parser.add_argument("--exit-threshold", dest="exit_threshold", type=str, default="auto",
                        help="Exit threshold for adaptive compute. Pass `none` to disable adaptive compute.")
    parser.add_argument("--num-steps", dest="num_steps", type=int, default=32, help="Number of steps for generation")
    parser.add_argument("--lookup-strategy", dest="lookup_strategy", type=str, default="full", 
                        help="Lookup strategy for caching, also supports values like `compression-s4`")
    parser.add_argument("--continuous-compute", dest="continuous_compute", type=bool, default=False, help="Continuous compute")
    parser.add_argument("--latent-dampening", dest="latent_dampening", type=bool, default=False, help="Latent dampening")
    parser.add_argument("--num_samples", dest="num_samples", type=int, default=20, help="number of samples")
    parser.add_argument("--seq_len", dest="seq_len", type=int, default=32, help="sequence length for evaluating the latent representations")
    parser.add_argument("--embed_dim", dest="embed_dim", type=int, default=5280, help="embedding dimensionality")
    parser.add_argument("--intermediate_size", dest="intermediate_size", type=int, default=17920, help="intermediate_size")
    parser.add_argument("--num_head", dest="num_head", type=int, default=55, help="number of attention head")
    parser.add_argument("--num_layer", dest="num_layer", type=int, default=2, help="number of attention layer")
    parser.add_argument("--correctness_threshold", dest="correctness_threshold", type=float, default=0.6, help="threshold probability for determining the correctness of each sample")
    args = parser.parse_args()
    
    #results = evaluate_single_task(
    #    task_name=args.task_name,
    #    model_name=args.model_name,
    #    device=args.device,
    #    batch_size=args.batch_size,
    #    num_fewshot=args.num_fewshot,
    #    limit=args.limit,
    #    criterion="none",
    #    exit_threshold="none",
    #    num_steps=args.num_steps,
    #    lookup_strategy=args.lookup_strategy,
    #    continuous_compute=args.continuous_compute,
    #    latent_dampening=args.latent_dampening,
    #)
    
    if "test" in args.stage:
        evaluate_with_resampling(args)
    else:
        evaluate_with_system_prompt(args)

