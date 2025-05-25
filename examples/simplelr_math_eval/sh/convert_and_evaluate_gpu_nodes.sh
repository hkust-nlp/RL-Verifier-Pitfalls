#!/bin/bash
set -x

# 参数检查
if [ "$#" -gt 16 ]; then
    echo "Usage: $0 <eval_script_path> <base_checkpoint_path> <init_model_path> <template> [benchmarks] [temperature] [max_tokens] [top_p] [tp_size] [ckpt_list_file] [output_dir] [overwrite] [n_sampling] [convert_model] [split] [n_test_sample]"
    exit 1
fi

# 获取参数
eval_script_path=$1
base_checkpoint_path=$2
init_model_path=$3
template=$4
benchmarks=$5
temperature=$6
max_tokens=$7
top_p=$8
tp_size=${9:-1}  # 如果未提供第5个参数，默认为1
ckpt_list_file=${10:-""}  # Optional parameter for checkpoint list
output_dir=${11:-"eval_results"}
overwrite=${12:-false}
n_sampling=${13:-1}
convert_model=${14:-false}  # New parameter for model conversion
split=${15:-"test"}
n_test_sample=${16:--1}
actor_dir="actor"

# 获取可用的GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
NUM_GPU_GROUPS=$((NUM_GPUS / tp_size))  # 计算可用的GPU组数

# 函数：复制 tokenizer 文件
copy_tokenizer_files() {
    local ckpt_path=$1
    local init_model_path=$2
    local files_to_copy=(
        "added_tokens.json"
        # "config.json"
        # "generation_config.json"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )
    if [ -f "$init_model_path/merges.txt" ]; then
        files_to_copy+=("merges.txt")
    fi
    # 创建目标路径，确保它存在
    if [ ! -d "$ckpt_path" ]; then
        mkdir -p "$ckpt_path"
        echo "Created checkpoint directory: $ckpt_path" >&2
    else
        echo "Checkpoint directory already exists: $ckpt_path" >&2
    fi

    # 复制每个文件
    for filename in "${files_to_copy[@]}"; do
        src="$init_model_path/$filename"
        dst="$ckpt_path/$filename"
        if [ -e "$src" ]; then
            cp "$src" "$dst"
            echo "Copied $src to $dst"
        else
            echo "Warning: $src does not exist."
        fi
    done
}

# 函数：转换单个检查点模型
convert_checkpoint() {
    local step_tag=$1
    local base_path=$2
    local hf_model_path=$3
    local gpu_ids=$4
    
    # Skip conversion for step 0 as it's directly copied from the HF checkpoint
    if [ "$step_tag" = "global_step_0" ]; then
        echo "Skipping conversion for step 0 (already copied from HF model)"
        return 0
    fi
    
    # Extract step number from step_tag (remove "global_step_" prefix)
    local step=${step_tag#global_step_}
    
    echo "Converting model for step $step..."
    
    # Create paths
    local actor_path="$base_path/$step_tag/$actor_dir"
    local target_dir="$actor_path/huggingface"
    local convert_done_file="$target_dir/.convert_done"
    
    # Skip if already converted
    if [ -f "$convert_done_file" ]; then
        echo "Model for step $step already converted. Skipping conversion."
        return 0
    fi
    
    # Create directory if it doesn't exist
    mkdir -p "$target_dir"
    
    # Run conversion
    CUDA_VISIBLE_DEVICES=$gpu_ids python3 sh/model_merger.py \
        --backend fsdp \
        --hf_model_path "$hf_model_path" \
        --local_dir "$actor_path" \
        --target_dir "$target_dir"
    
    # Create marker file if conversion successful
    if [ $? -eq 0 ]; then
        copy_tokenizer_files "$target_dir" "$init_model_path"
        touch "$convert_done_file"
        echo "Completed conversion for step $step"
        return 0
    else
        echo "Error converting model for step $step"
        return 1
    fi
}

# 函数：获取所有需要评估的检查点，并过滤掉已评估的
get_checkpoints_to_evaluate() {
    local base_path="$1"
    
    if [ -n "$ckpt_list_file" ] && [ -f "$ckpt_list_file" ]; then
        # Read checkpoints from the provided file
        cat "$ckpt_list_file"
    else
        # Original logic for getting all checkpoints
        local checkpoints=()
        for ckpt_dir in "$base_path"/global_step_*; do
            if [ -d "$ckpt_dir" ]; then
                step_tag=$(basename "$ckpt_dir")
                checkpoints+=("$step_tag")
            fi
        done
        
        if [ ${#checkpoints[@]} -eq 0 ]; then
            echo ""
        else
            printf "%s\n" "${checkpoints[@]}"
        fi
    fi
}

# 函数：在指定GPU上处理单个检查点
process_checkpoint() {
    local step_tag=$1
    local group_id=$2
    
    # 计算该组的GPU ID范围
    local start_gpu=$((group_id * tp_size))
    local gpu_ids=""
    for ((i=0; i<tp_size; i++)); do
        if [ -n "$gpu_ids" ]; then
            gpu_ids="${gpu_ids},"
        fi
        gpu_ids="${gpu_ids}$((start_gpu + i))"
    done
    
    ckpt_path="$base_checkpoint_path/$step_tag/$actor_dir/huggingface"
    
    # Convert model if needed
    if [ "$convert_model" = "true" ]; then
        convert_checkpoint "$step_tag" "$base_checkpoint_path" "$init_model_path" "$gpu_ids"
    fi
    
    echo "Evaluating checkpoint $step_tag on GPUs $gpu_ids" >&2
    
    output_path_new="$base_checkpoint_path/$output_dir/$step_tag"
    mkdir -p "$output_path_new"
    
    CUDA_VISIBLE_DEVICES=$gpu_ids bash "$eval_script_path" ${template} "$ckpt_path" "$output_path_new" "$temperature" "$max_tokens" "$top_p" "$benchmarks" "$overwrite" "$n_sampling" "$split" "$n_test_sample"
}

# 记录当前工作目录
original_dir=$(pwd)

# 主脚本部分修改
# 获取需要评估的检查点
readarray -t checkpoints_to_evaluate < <(get_checkpoints_to_evaluate "$base_checkpoint_path")

if [ ${#checkpoints_to_evaluate[@]} -eq 0 ]; then
    echo "No new checkpoints to evaluate." >&2
    exit 0
fi

# 检查GPU数量是否满足tp_size要求
if [ $((NUM_GPUS % tp_size)) -ne 0 ]; then
    echo "Error: Number of available GPUs ($NUM_GPUS) is not divisible by tp_size ($tp_size)" >&2
    exit 1
fi

echo "Found ${#checkpoints_to_evaluate[@]} checkpoints to evaluate:" >&2
printf '%s\n' "${checkpoints_to_evaluate[@]}" >&2
total_checkpoints=${#checkpoints_to_evaluate[@]}
eval_count=0
# 并行处理检查点，按GPU组分配
for i in "${!checkpoints_to_evaluate[@]}"; do
    group_id=$((i % NUM_GPU_GROUPS))
    step_tag="${checkpoints_to_evaluate[i]}"
    
    # 在后台启动处理任务
    process_checkpoint "$step_tag" "$group_id" &
    
    # 每启动NUM_GPU_GROUPS个任务后等待它们完成
    if [ $(((i + 1) % NUM_GPU_GROUPS)) -eq 0 ]; then
        wait
    fi
    eval_count=$((eval_count + 1))
    echo "Evaluating $eval_count/$total_checkpoints checkpoints ..."
done

# 等待所有剩余的后台任务完成
wait

cd "$original_dir"
echo "All conversions and evaluations completed."