<div align="center">

# Pitfalls of Rule- and Model-based Verifiers: A Case Study on Mathematical Reasoning

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.18892)  [![Hugging Face](https://img.shields.io/badge/verifier_pitfalls-fcd022?style=for-the-badge&logo=Huggingface&logoColor=000)](https://huggingface.co/collections/hkust-nlp/rl-verifier-pitfalls-68329f54bd8fd397534bfe66)

</div>



This repo contains the resources for the paper "Pitfalls of Rule- and Model-based Verifiers: A Case Study on Mathematical Reasoning."

In this paper, we investigate the reliability of verification systems used in reinforcement learning for mathematical reasoning tasks. Specifically, we analyze the limitations of both rule-based and model-based verifiers, which are commonly used to provide reward signals in reinforcement learning with verifiable rewards (RLVR). We show that rule-based verifiers, while precise, often suffer from high false negative rates, especially as model outputs become more diverse or expressive. On the other hand, model-based verifiers, though more flexible and accurate in static settings, are vulnerable to reward hacking during dynamic RL training, where models exploit verifier weaknesses to gain unearned rewards.


<div align="center">
<img src="assets/overview.png" width="720" alt="overview">
</div>

> Evaluation accuracy (**Left**) and Reward (**Right**) using different verifiers during RL training. In the **Right**, the "Training rewards" are from the verifier, while the "Oracle rewards" are from GPT-4.

## Table of Contents

- [Overview](#verifying-mathematical-reasoning-for-reinforcement-learning-pitfalls-of-rule--and-model-based-verifiers)
- [Main Takeaways](#-main-takeaways)
- [When Good Verifiers Go Bad: Reward Hacking in RL Training](#-when-good-verifiers-go-bad-reward-hacking-in-rl-training)
  - [Probing Verifier Robustness](#️-probing-verifier-robustness)
  - [Verifier Vulnerability Analysis](#-verifier-vulnerability-analysis)
- [Model Checkpoints](#-model-checkpoints)
- [Quick Start for RL Training](#-quick-start-for-rl-training)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)


## 🚩 Main Takeaways
Our study reveals critical limitations in both rule-based and model-based verifiers for RL in mathematical reasoning:

* Rule-based verifiers achieve near-perfect precision but suffer from low recall. They fail to recognize mathematically correct answers expressed in varied formats — causing false negatives that hinder policy learning, especially when verifying advance models like R1-Distilled-Qwen. 

* Model-based verifiers significantly improve recall and flexibility but are prone to reward hacking during RL training, where policy models learn to exploit their weaknesses, as shown in the **Right** bottom of the figure above.


* A probe study using adversarial answer patterns reveals that most of the model-based verifiers are vulnerable to adversarial attacks, especially for generative verifiers (e.g., those using chain-of-thought reasoning). More will be discussed in the next section.


## 🧨 When Good Verifiers Go Bad: Reward Hacking in RL Training


Reward hacking arises when a policy model learns to exploit verifier weaknesses instead of genuinely improving reasoning. As shown in the figure above, we observed:

- **Sudden spikes in training rewards** not matched by oracle (GPT-4o) rewards, signaling that the model is optimizing for the **verifier's blind spots**, not true correctness.
- **Model collapse** after prolonged training with certain fine-tuned model-based verifiers (e.g., R1-Distill-Verifier-1.5B), where performance deteriorates despite apparent reward improvements.
- **Adversarial behavior**, where models exploit simple tokens (e.g., `{`) or gibberish text to bypass verification.

### 🛡️ Probing Verifier Robustness

We design and release a **"Hacking Dataset"** of 13 adversarial patterns (e.g., gibberish, HTML tags, empty symbols) to evaluate verifier robustness. The dataset is available at [rl-verifier-pitfalls_hacking_data](https://huggingface.co/datasets/hkust-nlp/rl-verifier-pitfalls_hacking_data).

Key findings:

- Most model-based verifiers are vulnerable to even the simplest hacking patterns, e.g., gibberish text and empty symbols.

- Generative verifiers (e.g., general-verifier) tend to be more vulnerable than discriminative ones and show notably higher attack success rates compared to discriminative ones (e.g., xVerify).

### 🧪 Verifier Vulnerability Analysis

The table below shows success rates (%) of representative hacking patterns against various verifiers. A lower success rate indicates better robustness to attacks (lower is better).

| **Verifier** | **Adversarial Prefixes** | **Answer Explanation** | **Empty Symbols** | **Gibberish** | **Html Markdown** | **Prompt Injection** |
|--------------|--------------------------|------------------------|-------------------|---------------|-------------------|----------------------|
| Qwen2.5-1.5B | 7.4 | 12.5 | 3.4 | 0.4 | 5.9 | 11.5 |
| Qwen2.5-Math-1.5B | 20.8 | 77.9 | 44.4 | 5.5 | 26.3 | 22.7 |
| DS-R1-Distill-Qwen-1.5B | 21.7 | 25.5 | 23.6 | 20.8 | 13.6 | 5.3 |
| Qwen2.5-7B | 1.9 | 7.6 | 8.3 | 0.0 | 11.5 | 0.2 |
| Qwen2.5-Math-7B | 30.2 | 61.6 | 29.7 | 9.8 | 18.7 | 35.2 |
| DS-R1-Distill-Qwen-7B | 1.5 | 42.9 | 22.7 | 1.1 | 14.9 | 6.4 |
| R1-Distill-Verifier-1.5B | 35.0 | 27.6 | 29.5 | 10.6 | 15.5 | 16.1 |
| xVerify-0.5B-I | 0.0 | 0.4 | 0.2 | 0.2 | 0.0 | 0.0 |
| xVerify-3B-Ia | 0.2 | 1.1 | 0.2 | 0.0 | 0.6 | 0.4 |
| General-Verifier | 22.1 | 28.5 | 5.9 | 18.1 | 7.2 | 3.6 |

> Note: "DS" denotes DeepSeek, and for Qwen series models, the "instruct" suffix is omitted for clarity. Full results for all patterns are available in the paper.




## 💾 Model Checkpoints
We are releasing our customized verifier, [R1-Distill-Verifier-1.5B](https://huggingface.co/hkust-nlp/R1-Distill-Verifier-1.5B), as part of our open-source effort.

Additionally, we are open-sourcing multiple model checkpoints trained with different verifier configurations. You can access them via the links below:

|Model|Verifier|Link|
|---|---|---|
|Qwen-2.5-7B-Verifier-HF|[HuggingFace Math Verifier (HF)](https://github.com/huggingface/Math-Verify)|[🤗](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-Verifier-HF)|
|Qwen-2.5-7B-Verifier-R1-Qwen-1.5B |HF + [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)|[🤗](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-Verifier-R1-Qwen-1.5B)|
|Qwen-2.5-7B-Verifier-R1-Verifier-1.5B |HF + [R1-Distill-Verifier-1.5B](https://huggingface.co/hkust-nlp/R1-Distill-Verifier-1.5B)|[🤗](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-Verifier-R1-Verifier-1.5B)|
|Qwen-2.5-7B-Verifier-general-verifier |HF + [general-verifier](https://huggingface.co/TIGER-Lab/general-verifier)|[🤗](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-Verifier-general-verifier)|

All these models are also in our [Huggingface Collection](https://huggingface.co/collections/hkust-nlp/rl-verifier-pitfalls-68329f54bd8fd397534bfe66). 


## 🚀 Quick Start for RL Training


### Installation

Our code is implemented based on [Verl](https://github.com/volcengine/verl). We provide basic environment setup for training as follows, which only support custom environment setup and [FSDP training](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html). 

```bash
conda create -n verl python==3.9
conda activate verl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e . 
```

To install from docker image or utilize Megatron-lm, please refer to [Verl's documentation](https://verl.readthedocs.io/en/v0.2.x/start/install.html).

### Training 

As described in our paper, we train using the [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) dataset with prompts from [SimpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason). The prepared dataset is available at [deepscaler_simplelr](https://huggingface.co/datasets/hkust-nlp/deepscaler_simplelr). We extend HybridEngine to support model-based verifiers, enabling GPU offloading during idle periods.

The training leverages GRPO with Ray and vLLM for acceleration. First, launch a Ray cluster:
```bash
# launch the master node of ray 
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8
```
Edit `setup_env.sh` to configure environment variables. Then use `train_grpo_math_tune_ray.sh` to start training.

Here are examples for running RL with different verifiers:

* Huggingface Verifier only: 
```bash
bash train_grpo_math_tune.sh --genrm_enable False  --dataset_name deepscaler_simplelr
```


* DeepSeek-R1-Distill-Qwen-1.5B as verifier (with HybridEngine):
Firstly download the model from huggingface repo [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), then: 
```bash
bash train_grpo_math_tune.sh  --genrm_enable True  --genrrm_prompt_type r1_wo_question --genrrm_model_name DeepSeek-R1-Distill-Qwen-1.5B --genrrm_temperature 0.6 --genrrm_top_p 0.95  --genrm_max_response_length 8192
```

* Our Customized Verifier R1-Distill-Verifier-1.5B:
Firstly download the verifier from huggingface repo [R1-Distill-Verifier-1.5B](https://huggingface.co/hkust-nlp/R1-Distill-Verifier-1.5B), then: 

```bash
bash train_grpo_math_tune.sh  --genrm_enable True  --genrrm_prompt_type r1_with_question --genrrm_model_name R1-Distill-Verifier-1.5B --genrrm_temperature 0.6 --genrrm_top_p 0.95  --genrm_max_response_length 8192
```


### Evaluate

We used [Qwen Math's codebase](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation) for evaluation, but for fairness considerations, we completely prohibited solving problems by calling code. The `eval_math_nodes.sh` script provides the full pipeline for evaluation, results collection, and analysis. To use it, you'll need to specify a few environment variables within the script, and then run it as shown below:

Example: 
```bash
bash eval_math_nodes.sh \
    --run_name verl_train_Qwen-2.5-7B_genrm_enableFalse_deepscaler_simplelr   \
    --init_model Qwen-2.5-7B \
    --template qwen-boxed  \
    --tp_size 1 \
    --add_step_0 true  \
    --temperature 1.0 \
    --top_p 0.7 \
    --max_tokens 16000 \
    --benchmarks aime24,amc23,math500,olympiadbench,gsm8k,minerva_math \
    --n_sampling 1 \
    --convert_model true
```



## Citation

If you find this work helpful, please consider citing us:

```bibtex
xxxx
```


## Acknowledgement
We build our reinforcement learning algorithm as an extension of [Verl](https://github.com/volcengine/verl). During training, we incorporate the [Huggingface Math Verifier](https://github.com/huggingface/Math-Verify). For inference, we utilize [vLLM](https://github.com/vllm-project/vllm), and our evaluation scripts are developed based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation). 

We would like to especially thank the developers of DeepSeek-R1 and Kimi-K1.5 for their innovations and valuable contributions to the open-source community.


