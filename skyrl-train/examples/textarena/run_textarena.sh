set -x

# Example training run on a mixed TextArena games dataset.
# Generate the dataset first:
#   uv run examples/textarena/textarena_dataset.py --output_dir $HOME/data/textarena
# Then launch training (requires at least 1 GPU and configured WANDB):
#   export WANDB_API_KEY=<your_key>
#   bash examples/textarena/run_textarena.sh

DATA_DIR="$HOME/data/textarena"

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.critic.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.num_inference_engines=1 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=4 \
  trainer.critic_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  generator.n_samples_per_prompt=1 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  environment.env_class=textarena \
  trainer.logger="wandb" \
  trainer.project_name="textarena" \
  trainer.run_name="textarena_example" \
  "$@"
