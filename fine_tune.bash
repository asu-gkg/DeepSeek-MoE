# DATA_PATH="open-r1/OpenR1-Math-220k"
OUTPUT_PATH="./output/deepseek-moe-16b-base-openr1-math-220k"
MODEL_PATH="deepseek-ai/deepseek-moe-16b-base"

cd finetune
deepspeed finetune.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --bf16 True \
    --use_lora True \
    --lora_rank 8 \
    --lora_alpha 32