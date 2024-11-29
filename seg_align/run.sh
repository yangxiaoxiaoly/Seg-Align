
python segmentation.py

#use gpt
python re_rank_gpt.py
#use llama
torchrun --nproc_per_node 1 re_rank_llama.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 8000 --max_batch_size 6

python llama_result_process.py
python get_hits.py
python get_prf.py
