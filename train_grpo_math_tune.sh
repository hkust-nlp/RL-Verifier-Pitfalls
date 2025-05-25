source setup_env.sh
export VLLM_ATTENTION_BACKEND=XFORMERS
# Default values
TRAIN_BATCH_SIZE=1024
# This is the sampling size of every validation set to accelerate the testing during training.
# It should be used along with n_val (default 8) and val_temperature (default 1.0)
VAL_SAMPLE_SIZE=100000
N_VAL=1
VAL_TEMPERATURE=0.0

MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=8192
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=256
# per GPU
PPO_MICRO_BATCH_SIZE=4
CLIP_RATIO=0.2
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
KL_LOSS_COEF=0.0
ENTROPY_COEFFIENT=0.0
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
MIN_P=0.0
TOP_P=1.0
TOP_K=-1
LOG_PROB_MICRO_BATCH_SIZE=20
ROLLOUT_N=8
KL_COEF=0.001
TOTAL_EPOCHS=200
DATASET_NAME=deepscaler_simplelr
ROLLOUT_GPU_MEMORY_UTIL=0.6
ACTOR_OPTIMIZER_OFFLOAD=False
ACTOR_PARAMETER_OFFLOAD=False
MODEL_NAME=Qwen-2.5-7B
SAVE_FREQ=10
TEST_FREQ=10
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
MICRO_ROLLOUT_BATCH_SIZE=1024
REMOVE_PREVIOUS_CKPT=False
APPLY_CHAT_TEMPLATE=False
REJECTION_SAMPLE=False
DUPL_ROLLOUT=True # enable this to fix this bug https://github.com/vllm-project/vllm/issues/14759


# genrm config
GENRM_ENABLE=True
GENRMM_MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B
GENRMM_MAX_PROMPT_LENGTH=1024
GENRMM_MAX_RESPONSE_LENGTH=8192
GENRMM_TEMPERATURE=0.6
GENRMM_TOP_P=0.95
GENRMM_PROMPT_TYPE="r1_wo_question"


OVERWRITE_RUN_NAME=""


generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local model_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --kl_loss_coef) suffix+="_klloss$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entcoef$2"; shift 2 ;;
      --clip_ratio) suffix+="_clipratio$2"; shift 2 ;;
      --clip_ratio_low) suffix+="_clipratiolow$2"; shift 2 ;;
      --clip_ratio_high) suffix+="_clipratiohigh$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --top_p) suffix+="_topp$2"; shift 2 ;;
      --top_k) suffix+="_topk$2"; shift 2 ;;
      --min_p) suffix+="_minp$2"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --kl_coef) suffix+="_klcoef$2"; shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) suffix+="_$2"; model_provided=true; shift 2 ;;
      --reword_function_type) suffix+="_reward_type$2"; shift 2 ;;
      --format_penalty_value) suffix+="_format_penalty$2"; shift 2 ;;
      --remove_clip) suffix+="_remove_clip$2"; shift 2 ;;
      --apply_chat_template) suffix+="_applychat$2"; shift 2 ;;
      --rejection_sample) suffix+="_rejection_sample$2"; shift 2 ;;
      --genrm_enable) suffix+="_genrm_enable$2"; shift 2 ;;
      --genrrm_model_name) suffix+="_genrrm$2"; shift 2 ;;
      --genrrm_prompt_type) suffix+="_genrrm_prompt_type$2"; shift 2 ;;
      --genrrm_temperature) suffix+="_genrrm_temp$2"; shift 2 ;;
      --genrrm_top_p) suffix+="_genrrm_topp$2"; shift 2 ;;
      --genrrm_only) suffix+="_genrrm_only$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi

  if [ "$model_provided" = false ]; then
    suffix+="_$MODEL_NAME"
  fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi
  
  echo "$suffix"
}

echo "Arguments received: $@"

# Generate a unique suffix based on the input arguments
SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"

echo "RUN_NAME: $RUN_NAME"
echo "LOG_FILE_PATH: $LOG_FILE_PATH"


# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --val_sample_size) VAL_SAMPLE_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
    --clip_ratio_low) CLIP_RATIO_LOW="$2"; shift 2 ;;
    --clip_ratio_high) CLIP_RATIO_HIGH="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top_p) TOP_P="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --min_p) MIN_P="$2"; shift 2 ;;
    --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --n_val) N_VAL="$2"; shift 2 ;;
    --val_temperature) VAL_TEMPERATURE="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --micro_rollout_batch_size) MICRO_ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --dupl_rollout) DUPL_ROLLOUT="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --actor_optimizer_offload) ACTOR_OPTIMIZER_OFFLOAD="$2"; shift 2 ;;
    --actor_parameter_offload) ACTOR_PARAMETER_OFFLOAD="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --genrm_enable) GENRM_ENABLE="$2"; shift 2 ;;
    --genrrm_model_name) GENRMM_MODEL_NAME="$2"; shift 2 ;;
    --genrrm_temperature) GENRMM_TEMPERATURE="$2"; shift 2 ;;
    --genrrm_top_p) GENRMM_TOP_P="$2"; shift 2 ;;
    --genrrm_prompt_type) GENRMM_PROMPT_TYPE="$2"; shift 2 ;;
    --genrm_max_prompt_length) GENRMM_MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --genrm_max_response_length) GENRMM_MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --remove_clip) REMOVE_CLIP="$2"; shift 2 ;;
    --remove_previous_ckpt) REMOVE_PREVIOUS_CKPT="$2"; shift 2 ;;
    --apply_chat_template) APPLY_CHAT_TEMPLATE="$2"; shift 2 ;;
    --rejection_sample) REJECTION_SAMPLE="$2"; shift 2 ;;
    --overwrite_run_name) OVERWRITE_RUN_NAME="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -n "$OVERWRITE_RUN_NAME" ]; then
    RUN_NAME="$OVERWRITE_RUN_NAME"
fi

echo "RUN_NAME: $RUN_NAME"
echo "LOG_FILE_PATH: $LOG_FILE_PATH"
echo "Training with the following parameters:" | tee -a $LOG_FILE_PATH
echo "Train Batch Size: $TRAIN_BATCH_SIZE" | tee -a $LOG_FILE_PATH
echo "Max Prompt Length: $MAX_PROMPT_LENGTH" | tee -a $LOG_FILE_PATH
echo "Max Response Length: $MAX_RESPONSE_LENGTH" | tee -a $LOG_FILE_PATH
echo "Learning Rate: $LEARNING_RATE" | tee -a $LOG_FILE_PATH
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE" | tee -a $LOG_FILE_PATH
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE" | tee -a $LOG_FILE_PATH
echo "KL Loss Coefficient: $KL_LOSS_COEF" | tee -a $LOG_FILE_PATH
echo "KL Loss Type: $KL_LOSS_TYPE" | tee -a $LOG_FILE_PATH
echo "Temperature: $TEMPERATURE" | tee -a $LOG_FILE_PATH
echo "Rollout N: $ROLLOUT_N" | tee -a $LOG_FILE_PATH
echo "KL Coefficient: $KL_COEF" | tee -a $LOG_FILE_PATH
echo "Total Epochs: $TOTAL_EPOCHS" | tee -a $LOG_FILE_PATH
echo "Dataset Name: $DATASET_NAME" | tee -a $LOG_FILE_PATH
echo "Model Name: $MODEL_NAME" | tee -a $LOG_FILE_PATH
echo "Remove Clip: $REMOVE_CLIP" | tee -a $LOG_FILE_PATH
echo "Genrm Enable: $GENRM_ENABLE" | tee -a $LOG_FILE_PATH
echo "Genrm Model Name: $GENRMM_MODEL_NAME" | tee -a $LOG_FILE_PATH
echo "Genrm Temperature: $GENRMM_TEMPERATURE" | tee -a $LOG_FILE_PATH
echo "Genrm Top P: $GENRMM_TOP_P" | tee -a $LOG_FILE_PATH
echo "Genrm Prompt Type: $GENRMM_PROMPT_TYPE" | tee -a $LOG_FILE_PATH

max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)
GENRMM_MAX_NUM_BATCHED_TOKENS=$(expr $GENRMM_MAX_PROMPT_LENGTH + $GENRMM_MAX_RESPONSE_LENGTH + 1000)
# Example of using the variables


if (( $(echo "$GENRMM_TEMPERATURE > 0" | bc -l) )); then
    GENRM_DO_SAMPLE=True
else
    GENRM_DO_SAMPLE=False
fi
PPO_MAX_TOKEN_LEN=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH)
sleep 3
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
    data.val_files=["$HDFS_DATA_PATH/$DATASET_NAME/test.parquet"] \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_sample_size=$VAL_SAMPLE_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.apply_chat_template=$APPLY_CHAT_TEMPLATE \
    actor_rollout_ref.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAMETER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_k=$TOP_K \
    actor_rollout_ref.rollout.min_p=$MIN_P \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.n_val=$N_VAL \
    actor_rollout_ref.rollout.val_temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.micro_rollout_batch_size=$MICRO_ROLLOUT_BATCH_SIZE \
    actor_rollout_ref.rollout.duplicate_prompts_for_sampling=$DUPL_ROLLOUT  \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    generative_reward.enable=$GENRM_ENABLE \
    generative_reward.prompt_type=$GENRMM_PROMPT_TYPE \
    generative_reward.model.path=$GENRMM_MODEL_PATH/$GENRMM_MODEL_NAME \
    generative_reward.model.use_remove_padding=False \
    generative_reward.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    generative_reward.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    generative_reward.rollout.temperature=$GENRMM_TEMPERATURE \
    generative_reward.rollout.top_p=$GENRMM_TOP_P \
    generative_reward.rollout.do_sample=$GENRM_DO_SAMPLE \
    generative_reward.rollout.prompt_length=$GENRMM_MAX_PROMPT_LENGTH \
    generative_reward.rollout.response_length=$GENRMM_MAX_RESPONSE_LENGTH \
    generative_reward.rollout.name=vllm \
    generative_reward.rollout.max_num_batched_tokens=$GENRMM_MAX_NUM_BATCHED_TOKENS \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    critic.ppo_micro_batch_size_per_gpu=4 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.rejection_sample=$REJECTION_SAMPLE \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=8 \
    +trainer.val_before_train=True \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.remove_clip=$REMOVE_CLIP \
    trainer.val_generations_to_log_to_wandb=64 \
    trainer.remove_previous_ckpt_in_save=$REMOVE_PREVIOUS_CKPT \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS