
deepspeed --master_port 20345 --include localhost:4,5,6,7 --module llama.cli.pretrain \
    --model_name_or_path /processing_data/infra/lvjiahui/work/models/llama/llama3-8b \
    --flash_attn \
    --train_dataset /processing_data/infra/lvjiahui/work/datasets/medical/pretrain/medical_book_zh.json \
    --eval_dataset /processing_data/infra/lvjiahui/work/datasets/medical/pretrain/valid_encyclopedia.json \
    --input_key text \
    --max_len 1024 \
    --max_examples 1000000 \
    --output_dir ./output_dir \
    --zero_stage 2 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --save_steps 20 \
    --eval_steps 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 0.00005 \
    --num_train_epochs 3 \
    --swanlab_api_key 6IWs4PClMRoHWP0bU7Es2 \
    --swanlab_project_name llama \
    --swanlab_experiment_name pretrain