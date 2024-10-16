#!/bin/bash
PRETRAIN_NAME=MGM-7B-Pretrain
FINETUNE_NAME=MGM-7B
AUX_SIZE=768

# deepspeed mgm/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /data/vicuna/vicuna-7b-v1.5 \
#     --version plain \
#     --data_path /data/luogen_code/MGM/playground/data/modified_llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower /data/luogen_code/MGM/checkpoints/openai/clip-vit-large-patch14-336 \
#     --vision_tower_aux /data/luogen_code/MGM/checkpoints/openai/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_size_aux $AUX_SIZE \
#     --bf16 True \
#     --output_dir ./work_dirs/$PRETRAIN_NAME \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb


deepspeed mgm/train/train_mem.py \
    --capacity_factor 0.5 \
    --mod_enable True  \
    --mod_mode 'arank_mod' \
    --router_aux_loss_coef 0.01 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/vicuna/vicuna-7b-v1.5  \
    --version v1 \
    --data_path ./playground/data/mini_gemini_data/mgm_instruction.json \
    --image_folder ./playground/data/mini_gemini_data \
    --vision_tower model_zoo/OpenAI/clip-vit-large-patch14-336 \
    --vision_tower_aux model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    --pretrain_mm_mlp_adapter /data/luogen_code/MGM/checkpoints/MGM-Pretrain/MGM-7B/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --image_size_aux $AUX_SIZE \
    --bf16 True \
    --output_dir ./work_dirs/$FINETUNE_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True 