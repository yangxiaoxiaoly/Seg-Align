# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

import numpy as np
from Param import *
from tqdm import tqdm
import time
import torch
import transformers
import time
# from transformers import AutoTokenizer
# from langchain import LLMChain, HuggingFacePipeline, PromptTemplate


startime = time.time()

print("starttime:", startime)

# fold = "data/DBP15K/"
# lang = "zh_en/"
# k = 10
# f = str(k)

te = 0
print("candidate size:", f)


#加载测试集，将ent1 , ent2转化为字典
def load_ent(path):
    e1 = []
    e2 = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = tuple(line.strip().split('\t'))
            e1.append(t[0])
            e2.append(t[1])
    return e1, e2

def load_entid(path):
    e = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = tuple(line.strip().split('\t'))
            e[t[1]] = int(t[0])
    return e
def ent_to_dic(ent):
    ent_dic = {}
    i = 0
    for e in ent:
        ent_dic[i] = e.split("/")[-1]
        i += 1
    return ent_dic

def load_graph(path):
    graph_dic = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            t = line.strip().split("\t")
            graph_dic[int(t[0])] = t[1]
    return graph_dic

ent1, ent2 = load_ent(fold + lang + "test_links")

ent_dic_1 = ent_to_dic(ent1)
ent_dic_2 = ent_to_dic(ent2)

top_1 = np.load(fold + lang + 'top1.npy')
top_k = np.load(fold + lang + 'top'+ f + '.npy')

# re_rank_list = np.load(fold + lang + 're_rank_list.npy')
# re_rank_top10 = np.load(fold + lang + 're_rank_top10.npy')

hard_list = np.load(fold + lang + 'hard_list_'+ f + '.npy')
hard_topk = np.load(fold + lang + 'hard_top'+ f + '.npy')

simple_list = np.load(fold + lang + 'simple_list_'+ f + '.npy')
simple_topk = np.load(fold + lang + 'simple_top'+ f + '.npy')

# load neighbours
# graph1 = load_graph(fold + lang + "graph_1")
# graph2 = load_graph(fold + lang + "graph_2")



h_l = len(hard_list)
s_l = len(simple_list)
print(lang)
print("number of hard sample:", h_l)
print("number of simple sample:", s_l)

def write_data_2file(name, data):
    with open(name, 'a', encoding='UTF-8') as f:
        f.write(str(data) + "\n")


def main(    
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = te,
    top_p: float = 0.9,
    max_seq_len: Optional[int] = None,
    max_batch_size: int = 8,
    max_gen_len: int = 3,
    # max_gen_len: Optional[int] = None,
):
    # print("temperature: float = 0")
    
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    ent1_dic = ent_dic_1
    ent2_dic = ent_dic_2
    re_rank_list = hard_list
    re_rank_topk = hard_topk
    l = len(re_rank_list)
    
    for i in tqdm(range(l), desc="querying:"):
    
    
        source_ent = ent1_dic[re_rank_list[i]]
        # neibour1 = graph1[re_rank_list[i]]
        # print(len(neibour1))
        '''
        # add the neighbours
        letter_dic = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H",8:"I",9:"J"}
        target_ent = {}
        anwser = []
        for j in range(k):
            # print(len(graph2[re_rank_topk[i][j]]))
            target_ent[letter_dic[j]] = (ent2_dic[re_rank_topk[i][j]], graph2[re_rank_topk[i][j]])
            # target_ent[letter_dic[j]] += ent2_dic[re_rank_topk[i][j]]
            
            # if len(neibour1) >= 6000: 
            #     neibour1 = neibour1[:6000]
            # if len(graph2[re_rank_topk[i][j]]) >= 6000:
            #     # print(graph2[re_rank_topk[i][j]])
            #     graph2[re_rank_topk[i][j]] = graph2[re_rank_topk[i][j]][:6000]
            #     # print(graph2[re_rank_topk[i][j]])
            dialogs: List[Dialog] = [
                [                 

                 {"role": "system", "content": "Answer me 'Yes' or 'No'."},
                 # {"role": "user", "content": "This is source entity:" + source_ent + ". And this is the target entity:" + ent2_dic[re_rank_topk[i][j]] + ". Are the two entities the same entity?"}
                 {"role": "user", "content": "This is source entity:" + source_ent + ", and it's neibours:" + neibour1 + ". And this is the target entity:" + ent2_dic[re_rank_topk[i][j]] + ", and it's neibours:" + graph2[re_rank_topk[i][j]] + ". Are the two entities the same entity?"}
                 # {"role": "user", "content": "Choose the option that is most similar to entity: " + source_ent + ", from the follow options." + str(target_ent)}


                ]

            ]
            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )



            for dialog, result in zip(dialogs, results):
                for msg in dialog:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                    print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                    )
                print("\n==================================\n")

                # if len(f">{result['generation']['content']}".split('is:')) == 1:
                #     re = f">{result['generation']['content']}".split('is:')[0].strip()
                # else:
                #     re = f">{result['generation']['content']}".split('is:')[1].strip()

                re = f">{result['generation']['content']}".split("\n")
                # re_n = [r for r in re if len(r)!= 0]
                # print(f">{result['generation']['content']}".split("\n"))
                # print("\n==================================\n")
                # print(re)
                
                anwser.append(re)
            
        '''
        # print(str(target_ent))
        # without neighbours
        target_ent = str({"A": ent2_dic[re_rank_topk[i][0]],"B": ent2_dic[re_rank_topk[i][1]],"C": ent2_dic[re_rank_topk[i][2]],"D": ent2_dic[re_rank_topk[i][3]],"E": ent2_dic[re_rank_topk[i][4]],"F": ent2_dic[re_rank_topk[i][5]],"G": ent2_dic[re_rank_topk[i][6]],"H": ent2_dic[re_rank_topk[i][7]],"I": ent2_dic[re_rank_topk[i][8]],"J": ent2_dic[re_rank_topk[i][9]]})
        # target_ent = str({"A": ent2_dic[re_rank_topk[i][0]],"B": ent2_dic[re_rank_topk[i][1]],"C": ent2_dic[re_rank_topk[i][2]],"D": ent2_dic[re_rank_topk[i][3]],"E": ent2_dic[re_rank_topk[i][4]]})
        # target_ent = str({"A": ent2_dic[re_rank_topk[i][0]],"B": ent2_dic[re_rank_topk[i][1]],"C": ent2_dic[re_rank_topk[i][2]],"D": ent2_dic[re_rank_topk[i][3]],"E": ent2_dic[re_rank_topk[i][4]],"F": ent2_dic[re_rank_topk[i][5]],"G": ent2_dic[re_rank_topk[i][6]],"H": ent2_dic[re_rank_topk[i][7]],"I": ent2_dic[re_rank_topk[i][8]],"J": ent2_dic[re_rank_topk[i][9]],"K": ent2_dic[re_rank_topk[i][10]],"L": ent2_dic[re_rank_topk[i][11]],"M": ent2_dic[re_rank_topk[i][12]],"N": ent2_dic[re_rank_topk[i][13]],"O": ent2_dic[re_rank_topk[i][14]],"P": ent2_dic[re_rank_topk[i][15]],"Q": ent2_dic[re_rank_topk[i][16]],"R": ent2_dic[re_rank_topk[i][17]],"S": ent2_dic[re_rank_topk[i][18]],"T": ent2_dic[re_rank_topk[i][19]]})


        dialogs: List[Dialog] = [
            [
            {"role": "system", "content": "Answer me begin with 'The option is:'."},
            {"role": "user",
             "content": "Choose the option that is most similar to entity: " + source_ent + ", from the follow options." + target_ent}

            ]



        ]
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )



        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
            print("\n==================================\n")

            # if len(f">{result['generation']['content']}".split('is:')) == 1:
            #     re = f">{result['generation']['content']}".split('is:')[0].strip()
            # else:
            #     re = f">{result['generation']['content']}".split('is:')[1].strip()

            re = f">{result['generation']['content']}".split("\n")
            # re_n = [r for r in re if len(r)!= 0]
            # print(f">{result['generation']['content']}".split("\n"))
            # print("\n==================================\n")
            # print(re)

            # anwser.append(re)
            # print(anwser)


        if len(re_rank_list) > 5000:
            write_data_2file(fold + lang + "llama8b_simple_result_id" + f, re)
        else:
            write_data_2file(fold + lang + "llama8b_hard_result_id" + f, re)
                # print(f">{result['generation']['content']}".split(' ')[-1])
    #             print(f">{result['generation']['content']}")

    #             ans = f">{result['generation']['content']}".split('is')[1]

    #             print(str(ans))

        
if __name__ == "__main__":
    
    fire.Fire(main) 
    endtime = time.time()
    
    all_time = endtime - startime
    
    print("endtime:", endtime)
    print("time:", all_time)
    
    
    with open('code_time', 'a', encoding='UTF-8') as f1:
        f1.write(lang + "\t" + str(all_time) + "\t" + str(f) +"\n")
        
        
    
    
    # def query(l, ent1_dic, ent2_dic, re_rank_list, re_rank_topk):
    #     for i in tqdm(range(10), desc="询问中："):
    #         fire.Fire(main)  
    # # query(s_l, ent_dic_1, ent_dic_2, simple_list, simple_topk)
    # query(h_l, ent_dic_1, ent_dic_2, hard_list, hard_topk)   
    
    
         
        
        
# def query(l, ent1_dic, ent2_dic, re_rank_list, re_rank_topk):
#     for i in tqdm(range(10), desc="询问中："):
#         s_ent = ent1_dic[re_rank_list[i]]
#         t_ent = {"A": ent2_dic[re_rank_topk[i][0]],"B": ent2_dic[re_rank_topk[i][1]],"C": ent2_dic[re_rank_topk[i][2]],"D": ent2_dic[re_rank_topk[i][3]],"E": ent2_dic[re_rank_topk[i][4]],"F": ent2_dic[re_rank_topk[i][5]],"G": ent2_dic[re_rank_topk[i][6]],"H": ent2_dic[re_rank_topk[i][7]],"I": ent2_dic[re_rank_topk[i][8]],"J": ent2_dic[re_rank_topk[i][9]],}
#         main(s_ent, t_ent)
    
# # query(s_l, ent_dic_1, ent_dic_2, simple_list, simple_topk)
# query(h_l, ent_dic_1, ent_dic_2, hard_list, hard_topk)   
   
    
    
    
