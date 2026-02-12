# Latent Thinking Optimization 

This repo contains the code for Latent Thinking Optimization: Your Latent Reasoning Language Model Secretly Encodes Reward Signals in Its Latent Thoughts.

Our paper is accepted to ICLR 2026.

## Abstract

Large Language Models (LLMs) excel at problem solving by generating chain of thoughts in natural language, but such verbal thinking is computationally costly and prone to overthinking. Recent work instead proposes a latent thinking architecture Huginn-3.5B, which represents intermediate reasoning steps as sequence of latent representations. However, latent thoughts lack interpretability and are difficult to supervise, raising concerns about the correctness and reliability of its latent thinking processes. In this paper, we provide a systematic study of how Huginn-3.5B thinks in the latent space and how external supervision signals can improve its latent thinking processes. We show that latent thoughts leading to correct versus incorrect answers exhibit highly distinguishable patterns, and that a latent classifier can reliably predict answer correctness directly from latent thoughts. Leveraging these insights, we propose Latent Thinking Optimization (LTO), a probabilistic algorithm that employs the latent classifier as a Latent Reward Model (LRM) to optimize the latent thinking processes. Extensive experiments across diverse reasoning tasks demonstrate that LRM is highly effective in detecting incorrect latent thinking patterns, and LTO can significantly improve the latent thinking processes. Furthermore, we show that LRM can generalize across diverse domains, and LTO can be seamlessly applied to general LLMs to improve their thinking processes. In contrast to verbal thinking, our method demonstrates that reward modeling and scaling test-time thinking with supervision can be performed directly in the latent space, highlighting its potential as a general, efficient, and domain-agnostic approach to improving the thinking processes of LLMs.

## Latent Thinking Optimization for Huginn-3.5B

```
cd Huginn\evaluate_raven
```

Check `Huginn\evaluate_raven\README.MD` for detailed instructions

## Latent Thinking Optimization for General LLMs

```
cd GeneralLLMs
```

Check `GeneralLLMs\README.MD` for detailed instructions


## Environments
The environment requirements are in the file `environment.yml`.

## Latent Space Storage
The files to store the latent representations may take up a large space. Make sure you have enough space on your file system.