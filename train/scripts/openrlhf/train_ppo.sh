# checkpoint
save_path=./checkpoint/llama-3-8b-rlhf
save_steps=500
logging_steps=1
eval_steps=500

# deepspeed
micro_train_batch_size=8
micro_rollout_batch_size=4
rollout_batch_size=1024
train_batch_size=128
max_norm=1.0
zero_stage=2

# sft
max_epochs=1
pretrain=/processing_data/infra/lvjiahui/work/models/llama/llama3-8b
actor_learning_rate=5e-7
critic_learning_rate=9e-6
lr_warmup_ratio=0.0
l2=0.0 # weight decay


# dataset
prompt_data=json@/processing_data/infra/lvjiahui/work/datasets/ppo
dataset_probs=1.0 # 用来控制数据集大小
input_key=prompt
max_samples=300000 # all samples
prompt_max_len=1024 # cutoff length
generate_max_len=1024

# ppo
init_kl_coef=0.01

# wandb
use_wandb=a1582347264@gmail.com

deepspeed --include="localhost:4,5,6,7" --module openrlhf.cli.train_ppo \
   --pretrain /processing_data/infra/lvjiahui/8b/sft \
   --reward_pretrain /processing_data/infra/lvjiahui/8b/rm \
   --save_path ${save_path} \
   --save_steps ${save_steps} \
   --logging_steps ${logging_steps} \
   --eval_steps ${eval_steps} \
   --micro_train_batch_size ${micro_train_batch_size} \
   --train_batch_size ${train_batch_size} \
   --micro_rollout_batch_size ${micro_rollout_batch_size} \
   --rollout_batch_size ${rollout_batch_size} \
   --max_epochs ${max_epochs} \
   --prompt_max_len ${prompt_max_len} \
   --generate_max_len ${generate_max_len} \
   --zero_stage ${zero_stage} \
   --bf16 \
   --actor_learning_rate ${actor_learning_rate} \
   --critic_learning_rate ${critic_learning_rate} \
   --init_kl_coef ${init_kl_coef} \
   --prompt_data /processing_data/infra/lvjiahui/work/datasets/ppo \
   --input_key ${input_key} \
   --apply_chat_template \
   --max_samples ${max_samples} \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing

