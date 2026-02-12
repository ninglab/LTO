# Latent Thinking Optimization for General LLMs

# Generate Training Data for Latent Reward Models

## GSM8K

```
for i in $(seq 1 10)
do
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsm8k_train_$i --model_name allenai/OLMo-7B-0724-Instruct-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsm8k_train_$i --model_name meta-llama/Llama-2-7b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsm8k_train_$i --model_name meta-llama/Llama-2-13b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsm8k_train_$i --model_name mistralai/Mistral-7B-Instruct-v0.1
done
```

## SVAMP

```
for i in $(seq 1 10)
do
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset svamp_train_$i --model_name allenai/OLMo-7B-0724-Instruct-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset svamp_train_$i --model_name meta-llama/Llama-2-7b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset svamp_train_$i --model_name meta-llama/Llama-2-13b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset svamp_train_$i --model_name mistralai/Mistral-7B-Instruct-v0.1
done
```

## CommonsenseQA

```
for i in $(seq 1 10)
do
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset commonsenseqa_train_$i --model_name allenai/OLMo-7B-0724-Instruct-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset commonsenseqa_train_$i --model_name meta-llama/Llama-2-7b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset commonsenseqa_train_$i --model_name meta-llama/Llama-2-13b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset commonsenseqa_train_$i --model_name mistralai/Mistral-7B-Instruct-v0.1
done
```

## MBPP

```
for i in $(seq 1 50)
do
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset mbpp_train_$i --model_name allenai/OLMo-7B-0724-Instruct-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset mbpp_train_$i --model_name meta-llama/Llama-2-7b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset mbpp_train_$i --model_name meta-llama/Llama-2-13b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset mbpp_train_$i --model_name mistralai/Mistral-7B-Instruct-v0.1
done
```

# Sample Multiple Latent Thinking Trajectories for Latent Thinking Optimization

## GSM8K

```
for i in $(seq 1 20)
do
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsm8k_test_$i --model_name allenai/OLMo-7B-0724-Instruct-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsm8k_test_$i --model_name meta-llama/Llama-2-7b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsm8k_test_$i --model_name meta-llama/Llama-2-13b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsm8k_test_$i --model_name mistralai/Mistral-7B-Instruct-v0.1
done
```

## GSM-Symbolic

```
for i in $(seq 1 20)
do
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsmsymbolic_test_$i --model_name allenai/OLMo-7B-0724-Instruct-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsmsymbolic_test_$i --model_name meta-llama/Llama-2-7b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsmsymbolic_test_$i --model_name meta-llama/Llama-2-13b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset gsmsymbolic_test_$i --model_name mistralai/Mistral-7B-Instruct-v0.1
done
```

## SVAMP

```
for i in $(seq 1 20)
do
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset svamp_test_$i --model_name allenai/OLMo-7B-0724-Instruct-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset svamp_test_$i --model_name meta-llama/Llama-2-7b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset svamp_test_$i --model_name meta-llama/Llama-2-13b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset svamp_test_$i --model_name mistralai/Mistral-7B-Instruct-v0.1
done
```

## CommonsenseQA

```
for i in $(seq 1 20)
do
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset commonsenseqa_test_$i --model_name allenai/OLMo-7B-0724-Instruct-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset commonsenseqa_test_$i --model_name meta-llama/Llama-2-7b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset commonsenseqa_test_$i --model_name meta-llama/Llama-2-13b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset commonsenseqa_test_$i --model_name mistralai/Mistral-7B-Instruct-v0.1
done
```

## MBPP

```
for i in $(seq 1 20)
do
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset mbpp_test_$i --model_name allenai/OLMo-7B-0724-Instruct-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset mbpp_test_$i --model_name meta-llama/Llama-2-7b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset mbpp_test_$i --model_name meta-llama/Llama-2-13b-chat-hf
  python main.py --print_model_parameter --save_output --save_hidden_states --dataset mbpp_test_$i --model_name mistralai/Mistral-7B-Instruct-v0.1
done
```

# Train the Latent Reward Models

## GSM8K

```
python train_classifier_gsm8k_olmo_multiple_steps.py --classifier_file_save_path checkpoint_gsm8k_olmo.pth
python train_classifier_gsm8k_llama2_multiple_steps.py --classifier_file_save_path checkpoint_gsm8k_llama2.pth
python train_classifier_gsm8k_llama2_13b_multiple_steps.py --classifier_file_save_path checkpoint_gsm8k_llama2_13b.pth
python train_classifier_gsm8k_mistral_multiple_steps.py --classifier_file_save_path checkpoint_gsm8k_mistral.pth
```

## SVAMP

```
python train_classifier_gsm8k_olmo_multiple_steps.py --classifier_file_save_path checkpoint_svamp_olmo.pth
python train_classifier_gsm8k_llama2_multiple_steps.py --classifier_file_save_path checkpoint_svamp_llama2.pth
python train_classifier_gsm8k_llama2_13b_multiple_steps.py --classifier_file_save_path checkpoint_svamp_llama2_13b.pth
python train_classifier_gsm8k_mistral_multiple_steps.py --classifier_file_save_path checkpoint_svamp_mistral.pth
```

## CommonsenseQA

```
python train_classifier_gsm8k_olmo_multiple_steps.py --classifier_file_save_path checkpoint_commonsenseqa_olmo.pth
python train_classifier_gsm8k_llama2_multiple_steps.py --classifier_file_save_path checkpoint_commonsenseqa_llama2.pth
python train_classifier_gsm8k_llama2_13b_multiple_steps.py --classifier_file_save_path checkpoint_commonsenseqa_llama2_13b.pth
python train_classifier_gsm8k_mistral_multiple_steps.py --classifier_file_save_path checkpoint_commonsenseqa_mistral.pth
```

## MBPP

```
python train_classifier_gsm8k_olmo_multiple_steps.py --classifier_file_save_path checkpoint_mbpp_olmo.pth
python train_classifier_gsm8k_llama2_multiple_steps.py --classifier_file_save_path checkpoint_mbpp_llama2.pth
python train_classifier_gsm8k_llama2_13b_multiple_steps.py --classifier_file_save_path checkpoint_mbpp_llama2_13b.pth
python train_classifier_gsm8k_mistral_multiple_steps.py --classifier_file_save_path checkpoint_mbpp_mistral.pth
```

# Read the Correctness Rate of Latent Thinking Optimization

## GSM8K

```
python read_lrm_gsm8k_olmo.py
python read_lrm_gsm8k_llama2.py
python read_lrm_gsm8k_llama2_13b.py
python read_lrm_gsm8k_mistral.py
```

## GSM-Symbolic

```
python read_lrm_gsmsymbolic_olmo.py
python read_lrm_gsmsymbolic_llama2.py
python read_lrm_gsmsymbolic_llama2_13b.py
python read_lrm_gsmsymbolic_mistral.py
```

## SVAMP

```
python read_lrm_svamp_olmo.py
python read_lrm_svamp_llama2.py
python read_lrm_svamp_llama2_13b.py
python read_lrm_svamp_mistral.py
```

## CommonsenseQA

```
python read_lrm_commonsenseqa_olmo.py
python read_lrm_commonsenseqa_llama2.py
python read_lrm_commonsenseqa_llama2_13b.py
python read_lrm_commonsenseqa_mistral.py
```

## MBPP

```
python read_lrm_mbpp_olmo.py
python read_lrm_mbpp_llama2.py
python read_lrm_mbpp_llama2_13b.py
python read_lrm_mbpp_mistral.py
```



