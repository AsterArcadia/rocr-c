cd LLaMA-Factory
python run_expr_lora.py \
    --type=rocr_embed \
    --gpu=2 \
    --model=llama3 \
    --end_idx=1 

    python summarize.py --model=llama3 --type=rocr


python summarize.py --model=llama3 --type=rocr_embed

python run_expr_lora.py \
    --type=embed_proj \
    --gpu=0 \
    --model=llama3 \
    --start_idx=2 \
    --end_idx=10