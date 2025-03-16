#!/bin/bash
set -e  # Exit on error
set -x  # Print commands before execution

# ===== 环境设置 =====
# 设置工作目录
ROOT_PATH="~/GRPO_KKLogic"
# ===== 路径设置 =====
export ppl_num=2ppl,3ppl,4ppl,5ppl
EXPERIMENT_NAME='grpo_3b_kklogic_2345ppl'
# HDFS和本地路径
TRAIN_PATH="$ROOT_PATH/model_train/grpo/grpo_3b_kklogic_ckpt_2gpu_stage2_2345ppl"
MODEL_PATH="$ROOT_PATH/model_train/grpo/grpo_3b_stage1_3b_all_f1_a05_r0/actor/global_step_220"

# ===== WandB设置 =====
export WANDB_KEY='${WANDB_KEY}' # 替换为你的 WandB 密钥
wandb login $WANDB_KEY
export WANDB_PROJECT="grpo_kklogic"

cd $ROOT_PATH
# 安装依赖
pip install ray
pip install flash-attn --no-build-isolation
pip install -e .
pip install wandb IPython matplotlib
echo "==> pip环境安装成功"

# GPU配置
GPU_COUNT_PER_NODE=2
NODE_COUNT=1
GPU_MEMORY_UTILIZATION=0.7
TENSOR_PARALLEL_SIZE=2
ROLLOUT_N=4

# ===== 关键参数配置 =====
# 训练超参数
TEMPERATURE=1.0           # 模型采样温度
KL_COEF=0.003             # KL散度系数
LEARNING_RATE=5e-7        # 学习率
TOTAL_EPOCHS=3            # 总训练轮数
SAVE_FREQ=-1             # 保存频率
TEST_FREQ=-1             # 测试频率
CRITIC_WARMUP=30          # 评论家热身步数

# 批处理配置
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE=8
LOG_PROB_MICRO_BATCH_SIZE=64
MAX_PROMPT_LENGTH=400
MAX_RESPONSE_LENGTH=1024

# ===== 奖励配置 =====
#export TRAINING_STAGE=stage1
#export FORMAT_SCORE_BASE=1.0   # 降低格式分值权重
#export ANSWER_SCORE_BASE=0.5   # 轻微的答案激励
#export REASONING_BONUS_BASE=0.0 # 不考虑推理

# 设置 Stage 2
export TRAINING_STAGE=stage2
export FORMAT_SCORE_BASE=1.0      # 保持格式权重一致
export ANSWER_SCORE_BASE=1.0      # 增加答案权重但不过大
export REASONING_BONUS_BASE=0.5   # 适度的推理奖励

DATA_PATH="$ROOT_PATH/kk_data"
TMP_CHECKPOINT_DIR="$ROOT_PATH /tmp_checkpoints"

mkdir -p $TRAIN_PATH

# ===== 数据处理 =====
python3 "$ROOT_PATH/data_gen.py"
echo "==> ppl下载成功"

# ===== 环境变量 =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS

# ===== 执行训练 =====
# 运行训练脚本
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=$CRITIC_WARMUP \
    trainer.logger='["wandb"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$GPU_COUNT_PER_NODE \
    trainer.nnodes=$NODE_COUNT \
    trainer.default_local_dir=$TMP_CHECKPOINT_DIR \
    trainer.default_hdfs_dir=$TRAIN_PATH \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS "$@" 2>&1 | tee grpo.log