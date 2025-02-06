# checkpoint
save_path=path_to_save
save_steps=500
logging_steps=1
eval_steps=500

# deepspeed
micro_train_batch_size=8
train_batch_size=128
max_norm=1.0
zero_stage=2

# sft
max_epochs=3
pretrain=/processing_data/infra/lvjiahui/8b/sft
learning_rate=5e-5
lr_warmup_ratio=0.0
l2=0.0 # weight decay

# dataset
dataset=json@/processing_data/infra/lvjiahui/work/datasets/PKU-SafeRLHF-30K_openrlhf
dataset_probs=1.0 # 用来控制数据集大小
chosen_key=chosen
rejected_key=rejected
max_samples=10000000 # all samples
max_len=2048 # cutoff length

# wandb
use_wandb=a1582347264@gmail.com

deepspeed --module openrlhf.cli.train_rm \
   --save_path ${save_path} \
   --save_steps ${save_steps} \
   --logging_steps ${logging_steps} \
   --eval_steps ${eval_steps} \
   --micro_train_batch_size ${micro_train_batch_size} \
   --train_batch_size ${train_batch_size} \
   --max_norm ${max_norm} \
   --gradient_checkpointing \
   --zero_stage ${zero_stage} \
   --bf16 ${bf16} \
   --flash_attn \
   --max_epochs ${max_epochs} \
   --pretrain ${pretrain} \
   --learning_rate ${learning_rate} \
   --lr_warmup_ratio ${lr_warmup_ratio} \
   --l2 ${l2} \
   --dataset ${dataset} \
   --dataset_probs ${dataset_probs} \
   --chosen_key ${chosen_key} \
   --rejected_key ${rejected_key} \
   --max_samples ${max_samples} \
   --max_len ${max_len} \
   --use_wandb ${use_wandb} \
   --apply_chat_template \