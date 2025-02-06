# checkpoint
save_path=path_to_save
save_steps=500
logging_steps=2
eval_steps=500

# deepspeed
micro_train_batch_size=8
train_batch_size=128
max_norm=1.0
gradient_checkpointing=True
zero_stage=2

# sft
max_epochs=3
pretrain=/processing_data/infra/lvjiahui/work/models/llama/llama3-8b
learning_rate=5e-5
lr_warmup_ratio=0.0
l2=0.0 # weight decay

# lora
lora_rank=8
lora_alpha=64
target_modules=all-linear
lora_dropout=0.0

# dataset
dataset=json@/processing_data/infra/lvjiahui/study/LLM/RLHF/dataset/medical/pretrain
dataset_probs=1.0 # 用来控制数据集大小
input_key=text
max_samples=10000000 # all samples
max_len=4096 # cutoff length

# wandb
use_wandb=a1582347264@gmail.com
CUDA_VISIBLE_DEVICES=0,1,2,3

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} deepspeed --module openrlhf.cli.train_sft \
   --save_path ${save_path} \
   --save_steps ${save_steps} \
   --logging_steps ${logging_steps} \
   --eval_steps ${eval_steps} \
   --micro_train_batch_size ${micro_train_batch_size} \
   --train_batch_size ${train_batch_size} \
   --max_norm ${max_norm} \
   --gradient_checkpointing ${gradient_checkpointing} \
   --zero_stage ${zero_stage} \
   --bf16 \
   --flash_attn \
   --max_epochs ${max_epochs} \
   --pretrain ${pretrain} \
   --learning_rate ${learning_rate} \
   --lr_warmup_ratio ${lr_warmup_ratio} \
   --pretrain_mode \
   --l2 ${l2} \
   --lora_rank ${lora_rank} \
   --lora_alpha ${lora_alpha} \
   --target_modules ${target_modules} \
   --lora_dropout ${lora_dropout} \
   --packing_samples \
   --dataset ${dataset} \
   --dataset_probs ${dataset_probs} \
   --input_key ${input_key} \
   --max_samples ${max_samples} \
   --max_len ${max_len} \
   --use_wandb ${use_wandb}