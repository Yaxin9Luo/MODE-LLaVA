#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="mod-MGM-7B-HD-route_0.34_share_router"
SPLIT="llava_test_CQM-A"
SQADIR="./playground/data/eval/scienceqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m mgm.eval.model_vqa_science \
        --model-path ./work_dirs/$CKPT \
        --question-file $SQADIR/$SPLIT.json \
        --image-folder $SQADIR/images/test \
        --answers-file $SQADIR/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$SQADIR/answers/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $SQADIR/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python /data/luogen_code/MGM/mgm/eval/eval_science_qa.py \
    --base-dir $SQADIR \
    --result-file $output_file \
    --output-file $SQADIR/answers/output.jsonl \
    --output-result $SQADIR/answers/result.json
