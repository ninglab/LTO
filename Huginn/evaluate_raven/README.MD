# Latent Thinking Optimization for Huginn3.5B

# Generate Training Data for Latent Reward Models

## GSM8K

```
for i in $(seq 1 5)
do
  python hf_eval_adaptive_compute_gsm8k.py --stage train$i --num-steps 32 --dataset-path ./gsm8k/train.jsonl
done
```

## SVAMP

```
for i in $(seq 1 5)
do
  python hf_eval_adaptive_compute_svamp.py --stage train$i --num-steps 32 --dataset-path ./gsm8k/train.jsonl
done
```

## CommonsenseQA

```
for i in $(seq 1 5)
do
  python hf_eval_adaptive_compute_commonsenseqa.py --stage train$i --num-steps 32 --dataset-path ./commonsenseqa/train.jsonl
done
```

## MBPP

```
for i in $(seq 1 50)
do
  python hf_eval_adaptive_compute_mbpp.py --stage train$i --num-steps 32 --dataset-path ./commonsenseqa/train.jsonl
done
```

# Train the Latent Reward Models

## GSM8K

```
python train_classifier_gsm8k.py --classifier_file_save_path checkpoint_gsm8k.pth
```

## SVAMP

```
python train_classifier_svamp.py --classifier_file_save_path checkpoint_svamp.pth
```

## CommonsenseQA

```
python train_classifier_commonsenseqa.py --classifier_file_save_path checkpoint_commonsenseqa.pth
```

## MBPP

```
python train_classifier_mbpp.py --classifier_file_save_path checkpoint_mbpp.pth
```

# Sample Multiple Latent Thinking Trajectories for Latent Thinking Optimization

## GSM8K

```
python hf_eval_adaptive_compute_gsm8k.py --stage test --num-steps 32 --dataset-path ./gsm8k/test.jsonl
```
  
## GSM-Symbolic

```
python hf_eval_adaptive_compute_gsmsymbolic.py --stage test --num-steps 32 --dataset-path ./gsmsymbolic/GSM_symbolic.jsonl
```

## SVAMP

```
python hf_eval_adaptive_compute_svamp.py --stage test --num-steps 32 --dataset-path ./svamp/test.jsonl
```

## CommonsenseQA

```
python hf_eval_adaptive_compute_commonsenseqa.py --stage test --num-steps 32 --dataset-path ./commonsenseqa/test.jsonl
```

## MBPP

```
python hf_eval_adaptive_compute_mbpp.py --stage test --num-steps 32
```

# Read the Correctness Rate of Latent Thinking Optimization

## GSM8K

```
cd results/gsm8k
python read.py
```

## GSM-Symbolic

```
cd results/gsmsymbolic
python read.py
```

## SVAMP

```
cd results/svamp
python read.py
```

## CommonsenseQA

```
cd results/commonsenseqa
python read.py
```

## MBPP

```
cd results/mbpp
python read.py
```



