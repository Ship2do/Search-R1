#!/usr/bin/env bash
set -euo pipefail

# Select physical GPUs with CUDA_VISIBLE_DEVICES before invoking this script.
NUM_GPUS=${NUM_GPUS:-2}
DATA_DIR=${DATA_DIR:-data/nq_search}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-search-r1-learning-smoke}

case "$NUM_GPUS" in
    1|2|4|8) ;;
    *) echo "NUM_GPUS must be one of 1, 2, 4, or 8" >&2; exit 2 ;;
esac

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.max_start_length=768 \
    data.max_obs_length=128 \
    algorithm.adv_estimator=grpo \
    algorithm.no_think_rl=false \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size="$NUM_GPUS" \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$NUM_GPUS" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$NUM_GPUS" \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    trainer.project_name=Search-R1-learning \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    max_turns=1 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=3
