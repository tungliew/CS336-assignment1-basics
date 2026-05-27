#!/bin/bash

python Training.py \
    --vocab_size 50257 \
    --context_length 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --batch_size 32 \
    --T_w 2000 \
    --T_c 50000 \
    --max_iters 1000 \
    --rope_theta 10000.0 \
    --max_l2_norm 1.0 \
    --alpha_max 3e-4 \
    --alpha_min 3e-5 \
    --data_path /Users/tungliew/Downloads/assignment1-basics-main/data/TinyStoriesV2-GPT4-train-tokens.bin \
    --val_path /Users/tungliew/Downloads/assignment1-basics-main/data/TinyStoriesV2-GPT4-valid-tokens.bin \
    --device cpu \
    --checkpoint_path /Users/tungliew/Downloads/assignment1-basics-main/models/checkpoint.pt