# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from verl.utils.torch_functional import tokenize_and_postprocess_data
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
import numpy as np
import re
import random
from verl.utils.reward_score.qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
from verl.utils.reward_score.hf_math_verify import extract_solution as hf_extract_solution

def _default_postprocess_genrm_output(genrm_output: str, prompt_type: str) -> torch.Tensor:
    """
    Extract reward scores from generative reward model output text.
    The model is asked to determine mathematical equivalence and
    respond with either 1 or 0.
    
    Args:
        genrm_output: Text output from the generative reward model
        
    Returns:
        A scalar tensor with the extracted reward score
    """
    # Default score if nothing matches
    if prompt_type == "general-verifier":
        ext_re = r"Final Decision:\s*(yes|no|true|false)"
        match = re.search(ext_re, genrm_output, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).strip().lower()
            if extracted_answer.lower() in ["yes", "true"]:
                score = 1.0
            elif extracted_answer.lower() in ["no", "false"]:
                score = 0.0
            else:
                score = 0.0
        else:
            extracted_answer = ""
            score = 0.0
    else:
        score = 0.0
        extracted_answer = qwen_extract_answer(genrm_output, data_name="math")
        if extracted_answer.strip() == '1':
            score = 1.0
        elif extracted_answer.strip() == '0':
            score = 0.0
        else:
            score = 0.0
    return torch.tensor(score), extracted_answer


def is_numeric(value):
    if isinstance(value, (int, float)):
        return not isinstance(value, bool)
    elif isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False

def generate_genrm_prompt(prompt_type, ground_truth, extracted_answer, original_problem=None):
    """
    Generate prompt for generative reward model based on the specified prompt type.
    
    Args:
        prompt_type (str): Type of prompt to generate
        ground_truth (str): The ground truth answer
        extracted_answer (str): The extracted answer from model response
        original_problem (str, optional): The original problem text
        
    Returns:
        str: Formatted prompt for the generative reward model
    """
    deepseek_r1_system_prompt = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {input}\nAssistant:<think>\n'
    # original_problem
    if prompt_type == "r1_wo_question":
        # Original prompt without the question
        model_eval_prompt = '''
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Ground Truth Answer:**
{ground_truth}
**Extracted Answer:**
{extracted_answer}
- If **Extracted Answer** and **Ground Truth Answer** are mathematically equivalent, respond with \\boxed{{1}}
- If they are not mathematically equivalent, or if the **Extracted Answer** is nonsensical (e.g., a random string), respond with \\boxed{{0}}
'''
        prompt = model_eval_prompt.format(ground_truth=ground_truth, extracted_answer=extracted_answer)
        prompt = deepseek_r1_system_prompt.format(input=prompt)
    elif prompt_type == "r1_with_question":
        # Include the original question in the prompt
        model_eval_prompt = '''
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Question**
{original_problem}
**Ground Truth Answer:**
{ground_truth}
**Extracted Answer:**
{extracted_answer}
Please follow these steps clearly:
1. **Review the Question and Ground Truth Answer carefully.**
2. **Compare the Extracted Answer with the Ground Truth Answer.**
3. **Explain step-by-step** whether or not they express the same meaning or information.
4. **Provide your final decision clearly** at the end:
   - Respond with \\boxed{{1}} if the answers are equivalent.
   - Respond with \\boxed{{0}} if the answers are **not** equivalent.
'''
        prompt = model_eval_prompt.format(
            original_problem=original_problem,
            ground_truth=ground_truth, 
            extracted_answer=extracted_answer
        )
        prompt = deepseek_r1_system_prompt.format(input=prompt)
        
    elif prompt_type == "general-verifier":
        prompt = (
    f"User: ### Question: {original_problem}\n\n"
    f"### Ground Truth Answer: {ground_truth}\n\n"
    f"### Student Answer: {extracted_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)
    elif prompt_type == "qwen-boxed_wo_question":
        model_eval_prompt = '''
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Ground Truth Answer:**
{ground_truth}
**Extracted Answer:**
{extracted_answer}
- If **Extracted Answer** and **Ground Truth Answer** are mathematically equivalent, respond with \\boxed{{1}}
- If they are not mathematically equivalent, or if the **Extracted Answer** is nonsensical (e.g., a random string), respond with \\boxed{{0}}
'''
        prompt = model_eval_prompt.format(ground_truth=ground_truth, extracted_answer=extracted_answer)
        qwen_instruct_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" \
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" \
        "<|im_start|>assistant\n"
        
        prompt = qwen_instruct_template.format(input=prompt)
    elif prompt_type == "qwen-boxed_with_question":
        model_eval_prompt = '''
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Question**
{original_problem}
**Ground Truth Answer:**
{ground_truth}
**Extracted Answer:**
{extracted_answer}
Please follow these steps clearly:
1. **Review the Question and Ground Truth Answer carefully.**
2. **Compare the Extracted Answer with the Ground Truth Answer.**
3. **Explain step-by-step** whether or not they express the same meaning or information.
4. **Provide your final decision clearly** at the end:
   - Respond with \\boxed{{1}} if the answers are equivalent.
   - Respond with \\boxed{{0}} if the answers are **not** equivalent.
'''
        prompt = model_eval_prompt.format(
            original_problem=original_problem,
            ground_truth=ground_truth, 
            extracted_answer=extracted_answer
        )
        qwen_instruct_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" \
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" \
        "<|im_start|>assistant\n"
        
        prompt = qwen_instruct_template.format(input=prompt)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return prompt

class GenRMRewardManager:
    """The reward manager with two-stage evaluation: rule-based first, then generative model only for initially incorrect answers.
    """

    def __init__(self, config,  tokenizer,  num_examine, compute_score=None, postprocess_genrm_output_fn=None) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        local_path = copy_to_local(self.config.model.path)
        self.genrm_tokenizer = hf_tokenizer(local_path)
        self.max_prompt_length = self.config.rollout.prompt_length
        self.postprocess_genrm_output_fn = postprocess_genrm_output_fn or _default_postprocess_genrm_output
        self.prompt_type = self.config.get("prompt_type", "r1_wo_question")

        
    def rule_based_reward(self, data: DataProto, preprocess_for_genrm: bool = False) -> Union[torch.Tensor, dict]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32) # [batch_size]
        extra_info_dict: dict[str, list[float]] = {}
        extracted_answer_list: list[str] = []
        ground_truth_list: list[str] = []
        already_print_data_sources = {}
        response_list: list[str] = []   
        prompt_list: list[str] = []

        need_genrm_evaluation = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            prompt_list.append(prompt_str)
            response_list.append(response_str)
            
            
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            ground_truth_list.append(ground_truth)
            
            score_result = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score = score_result['score']
            if 'extra_info' in score_result:
                for key, value in score_result['extra_info'].items():
                    if key not in extra_info_dict:
                        extra_info_dict[key] = [0.0] * len(data)
                    extra_info_dict[key][i] = value
            extracted_answer_list.append(score_result['extracted_answer'])

            reward_tensor[i] = score

            # If the rule-based reward is 0 (incorrect), mark for genRM evaluation
            if score == 0 and extracted_answer_list[i] != '' and not (is_numeric(ground_truth_list[i]) and is_numeric(extracted_answer_list[i])):
                
                need_genrm_evaluation.append(i)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)                
        
        # Store which indices need genRM evaluation (scored 0 by rule-based reward)
        use_genrm_mask = torch.zeros(len(data), dtype=torch.bool)
        if need_genrm_evaluation:
            use_genrm_mask[need_genrm_evaluation] = True
            
        if preprocess_for_genrm:
            # Only process items that need genRM evaluation
            if need_genrm_evaluation:

                preprocessed_data = self.preprocess_for_genrm(data, extracted_answer_list, ground_truth_list, use_genrm_mask, prompt_list, response_list)
            else:
                # If no items need genRM, return empty batch
                preprocessed_data = None
        else:
            preprocessed_data = None
        
        if extra_info_dict or preprocess_for_genrm:
            return {
                'reward_tensor': reward_tensor, 
                'extra_info': extra_info_dict, 
                'genrm_batch': preprocessed_data
            }
        else:
            return reward_tensor
    
    
    def preprocess_for_genrm(self, data: DataProto, extracted_answer_list: list[str], ground_truth_list: list[str], mask_to_process: torch.Tensor = None, prompt_list: list[str] = None, response_list: list[str] = None) -> DataProto:
        """
        Preprocess data for generative reward model, optionally filtering to only include specified indices.
        
        Args:
            data: Input DataProto object
            extracted_answer_list: List of extracted answers
            ground_truth_list: List of ground truths
            mask_to_process: Optional list of indices to process (for filtering out already correct answers)
            prompt_list: List of original prompts (problems)
            response_list: List of responses from the actor model
        """
        formatted_prompts = []
        assert len(prompt_list) == len(response_list) == len(extracted_answer_list) == len(ground_truth_list) == mask_to_process.shape[0] == len(data)
        for i, (ground_truth, extracted_answer) in enumerate(zip(ground_truth_list, extracted_answer_list)):
            original_problem = data[i].non_tensor_batch['extra_info']['question']
            prompt = generate_genrm_prompt(
                self.prompt_type, 
                ground_truth, 
                extracted_answer, 
                original_problem
            )
            formatted_prompts.append(prompt)
        
        
        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        for prompt in formatted_prompts:
            input_ids, attention_mask = tokenize_and_postprocess_data(prompt=prompt, tokenizer=self.genrm_tokenizer, max_length=self.max_prompt_length, pad_token_id=self.genrm_tokenizer.pad_token_id, left_pad=True, truncation='right')
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids = compute_position_id_with_mask(attention_mask)
            position_ids_list.append(position_ids)
        
        attention_mask = torch.cat(attention_mask_list, dim=0)
        position_ids = torch.cat(position_ids_list, dim=0)
        input_ids = torch.cat(input_ids_list, dim=0)

        # Store the original indices to help with merging results later
        mask_to_process = mask_to_process if mask_to_process is not None else torch.ones(input_ids.shape[0], dtype=torch.bool)
        
        if not torch.all(mask_to_process):
            input_ids = input_ids[mask_to_process]
            attention_mask = attention_mask[mask_to_process]
            position_ids = position_ids[mask_to_process]
            extracted_answer_list = [extracted_answer_list[i] for i in range(len(extracted_answer_list)) if mask_to_process[i]]
            ground_truth_list = [ground_truth_list[i] for i in range(len(ground_truth_list)) if mask_to_process[i]]
            formatted_prompts = [formatted_prompts[i] for i in range(len(formatted_prompts)) if mask_to_process[i]]
            prompt_list = [prompt_list[i] for i in range(len(prompt_list)) if mask_to_process[i]]
            response_list = [response_list[i] for i in range(len(response_list)) if mask_to_process[i]]
            original_indices = torch.arange(len(data))[mask_to_process]
        else:
            original_indices = torch.arange(input_ids.shape[0])
            
        
        return_data = DataProto.from_single_dict({
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'position_ids': position_ids, 
            "extracted_answer": np.array(extracted_answer_list, dtype=object), 
            "ground_truth": np.array(ground_truth_list, dtype=object),
            "raw_input": np.array(formatted_prompts, dtype=object),
            "original_indices": np.array(original_indices, dtype=np.int32), # the indices of the original data
            "original_problems": np.array(prompt_list, dtype=object), # the original problems, here is the input of actor model 
            "original_responses": np.array(response_list, dtype=object) # the original responses, here is the response of actor model
        })

        return return_data
    
    def postprocess_genrm_output(self, input_batch:DataProto ,data: DataProto, init_reward_tensor: torch.Tensor = None, ) -> torch.Tensor:
        """Process the generative model outputs to extract reward scores"""
        # Initialize reward tensor for all responses in the batch
        reward_tensor = init_reward_tensor.clone() if init_reward_tensor is not None else torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        original_indices = input_batch.non_tensor_batch['original_indices']
        original_problems = input_batch.non_tensor_batch['original_problems']
        original_responses = input_batch.non_tensor_batch['original_responses']
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            idx = original_indices[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            
            prompt_str = self.genrm_tokenizer.decode(valid_prompt_ids)
            response_str = self.genrm_tokenizer.decode(valid_response_ids)
            sequences_str = prompt_str + response_str

            score, extracted_answer = self.postprocess_genrm_output_fn(response_str, self.prompt_type)
            reward_tensor[idx] = score
            
            if random.random() < 0.05:
                print(f"\n[Original Problem]\n{original_problems[i]}")
                print(f"\n[Actor Response]\n{original_responses[i]}")
                print(f"\n[GenRM Response]\n{sequences_str}")
                print(f"\n[GenRM Extracted Answer]\n{extracted_answer}")
                print(f"\n[GenRM Reward Score]\n{reward_tensor[idx]}")
        return reward_tensor
    
    def pad_for_distributed_inference(self, data: DataProto, world_size: int) -> tuple[DataProto, torch.Tensor]:
        """
        Pad the input to make it divisible by world_size.
        
        Args:
            data: Input DataProto object
            world_size: World size of the distributed engine
            
        Returns:
            Tuple of (padded DataProto, original_mask) where original_mask 
            indicates which items are from the original batch
        """
        batch_size = len(data)
        
        # If already divisible, no padding needed
        if batch_size % world_size == 0:
            return data, torch.ones(batch_size, dtype=torch.bool)
            
        # Calculate padding needed
        padding_size = world_size - (batch_size % world_size)
        original_mask = torch.ones(batch_size + padding_size, dtype=torch.bool)
        original_mask[batch_size:] = 0  # Mark padding items as False
        
        # Create copies of the last item for padding
        padded_batch = {}
        non_tensor_padded_batch = {}
        
        # Process tensor batch items
        for key, tensor in data.batch.items():
            # Get the last item and repeat it for padding
            last_item = tensor[-1:].expand(padding_size, *tensor.shape[1:])
            padded_batch[key] = torch.cat([tensor, last_item], dim=0)
            
        # Process non-tensor batch items
        for key, value in data.non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                last_item = value[-1:]
                # For object arrays, we need to handle differently
                if value.dtype == np.dtype('O'):
                    padding = np.array([last_item[0]] * padding_size, dtype=np.dtype('O'))
                else:
                    padding = np.repeat(last_item, padding_size, axis=0)
                non_tensor_padded_batch[key] = np.concatenate([value, padding], axis=0)
            else:
                # For other types, create a simple list
                non_tensor_padded_batch[key] = list(value) + [value[-1]] * padding_size
                
        # Create new DataProto with padding
        padded_data = DataProto.from_single_dict({**padded_batch, **non_tensor_padded_batch})
        
        return padded_data, original_mask
    
    def strip_padding(self, data: DataProto, original_mask: torch.Tensor) -> DataProto:
        """
        Strip the padding from the data based on the original mask.
        
        Args:
            data: DataProto object with padding
            original_mask: Boolean tensor indicating which items are from the original batch
            
        Returns:
            DataProto with padding removed
        """
        # Process tensor batch items
        filtered_batch = {
            key: value[original_mask] if isinstance(value, torch.Tensor) else value[original_mask.numpy()] 
            for key, value in data.batch.items()
        }
        
        # Process non-tensor batch items
        filtered_non_tensor_batch = {}
        for key, value in data.non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                filtered_non_tensor_batch[key] = value[original_mask.numpy()]
            else:
                # For other types, filter based on mask
                mask_list = original_mask.tolist()
                filtered_non_tensor_batch[key] = [item for i, item in enumerate(value) if mask_list[i]]
        
        # Create new DataProto without padding
        return DataProto.from_single_dict({**filtered_batch, **filtered_non_tensor_batch})
    
    
    def __call__(self, data: DataProto, genrm_engine) -> Union[torch.Tensor, dict]:
        """Two-stage reward evaluation process"""
        # First, calculate rule-based rewards for all responses
        rule_result = self.rule_based_reward(data, preprocess_for_genrm=True)
        
        # If rule_result is just a tensor, return it
        if not isinstance(rule_result, dict):
            return rule_result
        genrm_batch = rule_result['genrm_batch']
        
        # Get the rule-based reward tensor and other info
        rule_reward_tensor = rule_result['reward_tensor']
        
        # If no responses need genRM evaluation, return rule-based rewards
        if genrm_batch is None or len(genrm_batch) == 0:
            rule_reward_tensor = self.convert_reward_to_token_level_tensor(data, rule_reward_tensor)
            if 'extra_info' in rule_result:
                return {'reward_tensor': rule_reward_tensor, 'extra_info': rule_result['extra_info']}
            else:
                return rule_reward_tensor
                
        # Process only incorrect responses with genRM
        genrm_input = genrm_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        
        # Pad the input for distributed inference
        world_size = getattr(genrm_engine, 'world_size', 1)  # Default to 1 if not defined
        # breakpoint()
        padded_genrm_input, original_mask = self.pad_for_distributed_inference(genrm_input, world_size)
        
        # Generate sequences with the padded input
        padded_genrm_output = genrm_engine.generate_sequences(padded_genrm_input)
        
        # Remove the padding from the output using the strip_padding method
        genrm_output = self.strip_padding(padded_genrm_output, original_mask)
        # genrm_batch.union(genrm_output)
        genrm_reward = self.postprocess_genrm_output(genrm_batch, genrm_output, init_reward_tensor=rule_reward_tensor, 
                                                    )
        
        token_level_reward_tensor = self.convert_reward_to_token_level_tensor(data, genrm_reward)
            
        if 'extra_info' in rule_result:
            return {'reward_tensor': token_level_reward_tensor, 'extra_info': rule_result['extra_info']}
        else:
            return token_level_reward_tensor
        
    def convert_reward_to_token_level_tensor(self, data: DataProto, reward_tensor: torch.Tensor) -> torch.Tensor:
        """Convert the reward to a token-level tensor"""
        token_level_reward_tensor =  torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            token_level_reward_tensor[i, valid_response_length - 1] = reward_tensor[i]
        return token_level_reward_tensor
    
    
