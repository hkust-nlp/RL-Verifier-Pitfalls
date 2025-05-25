<div align="center">

# Verifying Mathematical Reasoning for Reinforcement Learning: Pitfalls of Rule- and Model-based Verifiers

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.18892)  [![Hugging Face](https://img.shields.io/badge/verifier_pitfalls-fcd022?style=for-the-badge&logo=Huggingface&logoColor=000)](https://huggingface.co/collections/hkust-nlp/xxxxx)

</div>



This repo contains the resources for the paper "Verifying Mathematical Reasoning for Reinforcement Learning: Pitfalls of Rule- and Model-based Verifiers". 

In this paper, we investigate the reliability of verification systems used in reinforcement learning for mathematical reasoning tasks. Specifically, we analyze the limitations of both rule-based and model-based verifiers, which are commonly used to provide reward signals in reinforcement learning with verifiable rewards (RLVR). We show that rule-based verifiers, while precise, often suffer from high false negative rates, especially as model outputs become more diverse or expressive. On the other hand, model-based verifiers, though more flexible and accurate in static settings, are vulnerable to reward hacking during dynamic RL training, where models exploit verifier weaknesses to gain unearned rewards.


<div align="center">
<img src="assets/overview.png" width="720" alt="overview">
</div>

> Evaluation accuracy (**Left**) and Reward (**Right**) using different verifiers during RL training. In the **Right**, the "Training rewards" are from the verifier, while the "Oracle rewards" are from GPT-4.



# 🚩 Main Takeaways
Our study reveals critical limitations in both rule-based and model-based verifiers for RL in mathematical reasoning:

* Rule-based verifiers achieve near-perfect precision but suffer from low recall. They fail to recognize mathematically correct answers expressed in varied formats — causing false negatives that hinder policy learning, especially when verifying advance models like R1-Distilled-Qwen. 

* Model-based verifiers significantly improve recall and flexibility but are prone to reward hacking during RL training, where policy models learn to exploit their weaknesses, as shown in the **Right** bottom of the figure above.


* A probe study using adversarial answer patterns reveals that most of the model-based verifiers are vulnerable to adversarial attacks, especially for generative verifiers (e.g., those using chain-of-thought reasoning). More will be discussed in the next section.


# 🧨 When Good Verifiers Go Bad: Reward Hacking in RL Training


Reward hacking arises when a policy model learns to exploit verifier weaknesses instead of genuinely improving reasoning. As shown in the figure above, we observed:

- **Sudden spikes in training rewards** not matched by oracle (GPT-4o) rewards, signaling that the model is optimizing for the **verifier's blind spots**, not true correctness.
- **Model collapse** after prolonged training with certain fine-tuned model-based verifiers (e.g., R1-Distill-Verifier-1.5B), where performance deteriorates despite apparent reward improvements.
- **Adversarial behavior**, where models exploit simple tokens (e.g., `{`) or gibberish text to bypass verification.

## 🛡️ Probing Verifier Robustness

We design and release a **"Hacking Dataset"** of 13+ adversarial patterns (e.g., gibberish, HTML tags, empty symbols) to evaluate verifier robustness. Key findings:

- Most model-based verifiers are vulnerable to even the simplest hacking patterns, e.g., gibberish text and empty symbols.

- Generative verifiers (e.g., general-verifier) tend to be more vulnerable than discriminative ones and show notably higher attack success rates compared to discriminative ones (e.g., xVerify).

## 🧪 Verifier Vulnerability Analysis

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




# Model Checkpoints

We are open-sourcing the model checkpoints trained with different verifiers. You can download them from the following links:


- [Qwen-2.5-7B-Verifier-HF](https://huggingface.co/xxxxxx) -- Trained with Huggingface Verifier
- [Qwen-2.5-7B-Verifier-R1-Qwen-1.5B](https://huggingface.co/xxxxxx) -- Trained with DeepSeek-R1-Distill-Qwen-1.5B as verifier 
- [Qwen-2.5-7B-Verifier-R1-Verifier-1.5B](https://huggingface.co/xxxxxx) -- Trained with R1-Distill-Verifier-1.5B as verifier 

And we also release the model checkpoints for our customized verifier R1-Distill-Verifier-1.5B, you can download it from the following link:

- [R1-Distill-Verifier-1.5B](https://huggingface.co/xxxxxx)


# Quick Start


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

As mentioned in our paper, our training data is from [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) with the prompt from [SimpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason). As mentioned in our paper, we extend HybridEngine to the model-based verifier, allowing it to be offloaded from GPUs during idle periods. 


The training process leverages GRPO with Ray and vLLM for acceleration. So firstly, you need to launch the ray cluster using the command below:
```bash
# launch the master node of ray 
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8
```
The main script for training is train_grpo_math_tune_ray.sh. But you need to specify the required environment variables in setup_env.sh. Once configured, submit the training job from the master node.


Here are examples for running RL with different verifiers:

* Huggingface Verifier only: 
```bash
bash start_ray.sh train_grpo_math_tune.sh --genrm_enable False  --dataset_name deepscaler_simplelr
```


* DeepSeek-R1-Distill-Qwen-1.5B  as verifier with HybridEngine: 
Firstly download the model from huggingface repo [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), then: 
```bash
bash start_ray.sh train_grpo_math_tune.sh  --genrm_enable True  --genrrm_prompt_type r1_wo_question --genrrm_model_name DeepSeek-R1-Distill-Qwen-1.5B --genrrm_temperature 0.6 --genrrm_top_p 0.95  --genrm_max_response_length 8192
```

* Our Customized Verifier R1-Distill-Verifier-1.5B:
Firstly download the model from huggingface repo [R1-Distill-Verifier-1.5B](https://huggingface.co/xxxxxx), then: 

```bash
bash start_ray.sh train_grpo_math_tune.sh  --genrm_enable True  --genrrm_prompt_type r1_with_question --genrrm_model_name R1-Distill-Verifier-1.5B --genrrm_temperature 0.6 --genrrm_top_p 0.95  --genrm_max_response_length 8192
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


# Citation

If you find this work useful, please consider citing:

```bibtex
xxxx
```


# Acknowledgement
We implement our reinforcement learning algorithm extending from [Verl](https://github.com/volcengine/verl). We utilize [vLLM](https://github.com/vllm-project/vllm) for inference and develop evaluation scripts based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation). Particularly, we thank the developers of DeepSeek-R1 and Kimi-k1.5 for their innovation and contribution to the open-source community.

