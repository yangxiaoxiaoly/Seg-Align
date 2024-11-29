# Seg-Align

This is code and datasets for Seg-Align

## Dependencies

1. Python 3.9
2. PyTorch 2.3.0
3. Numpy
4. Llama 3 and Llama 2
5. GPT-3.5 API 

## Dataset

All the data that used in our experiments are avaliable at https://drive.google.com/file/d/1zyz7Ur16KhOeULGWPxiDkTH1vjyPeZXn/view?usp=drive_link.

DBP15K and SRPRS are all from SDEA (https://github.com/zhongziyue/SDEA)

test_links, valid_links, test_emb_1.pt, test_emb_2.pt, valid_emb_1.pt, valid_emb_2.pt in the datasets are all from SDEA. (Run the SDEA code get the embeddings.) 


## Installation

Install llama3 and llama2 according to https://github.com/meta-llama/llama3 and https://github.com/meta-llama/llama.

## How to run

First segment the data.

```python
python segmentation.py
```

Then use LLM to process the hard samples.

```python
#use gpt
python re_rank_gpt.py
#use llama
torchrun --nproc_per_node 1 re_rank_llama.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 8000 --max_batch_size 6
```

Process the results obtained from llama.

```
python llama_result_process.py
```

Get the results.

```
python get_hits.py
python get_prf.py
```

