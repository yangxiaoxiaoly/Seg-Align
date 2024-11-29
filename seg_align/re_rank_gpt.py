# import torch
import numpy as np
import requests
from tqdm import tqdm
import time
import os
import openai
from Param import *
# fold = "data/DBP15K/"
# lang = "zh_en/"
# k = 10
# f = str(k)
print("candidate size:", f)
openai.api_key = 'your-key'

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



h_l = len(hard_list)
s_l = len(simple_list)
print(lang)
print("number of hard sample", h_l)
print("number of simple sample", s_l)


def get_completion(prompt, model="gpt-3.5"):
    messages = [{"role": "user", "content": prompt}]
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=messages,
    #     temperature=0,  # this is the degree of randomness of the model's output
    # )
    # #     return response.usage.completion_tokens
    url = "https://api.openai.com/v1/chat/completions"
    # replace your_key with actual key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {'your_key'}"
    }
    proxies = {
        # "http": "http://127.0.0.1:7890",
        # "https": "http://127.0.0.1:7890"
    }
    parameters = {
        "model": model,
        "messages": messages,
        # [
            # {
            #     "prompt": "Choose the option that is most similar to 马扎尔人 from the following options: A:Hungarians, B:Murad_I, C:North_America, D:Me_(Super_Junior-M_album), E:Zhongyuan, F:{ent2_dic[re_rank_top10[i][5]]}, G:Prokopis_Pavlopoulos, H:Johannes_Blaskowitz, I:Chiang_Pin-kung, J:Kim_Jong-nam.",
            #     "messages": "A"
            # }
        # ],
        "max_tokens": 1,
        "temperature": 0
    }
    response = requests.post(url, headers=headers, json=parameters, proxies=proxies)
    data = response.json()
    # print(data)
    text = data["choices"][0]["message"]['content']

    return text


def write_data_2file(name, data):
    with open(name, 'a', encoding='UTF-8') as f:
        f.write(str(data) + "\n")

def query(l, ent1_dic, ent2_dic, re_rank_list, re_rank_topk):
    for i in tqdm(range(l), desc="querying:"):
        # # 候选集5个
        # prompt = f"""
        #         Choose the option that is most similar to 马扎尔人 from the following options: A:Hungarians, B:Murad_I, C:North_America, D:Me_(Super_Junior-M_album).
        #         A
        #         Choose the option that is most similar to {ent1_dic[re_rank_list[i]]} from the following options: A:{ent2_dic[re_rank_topk[i][0]]}, B:{ent2_dic[re_rank_topk[i][1]]}, C:{ent2_dic[re_rank_topk[i][2]]},
        #                         D:{ent2_dic[re_rank_topk[i][3]]}, E:{ent2_dic[re_rank_topk[i][4]]}.
        #
        #         """
        # 候选集10个
        prompt = f"""
            Choose the option that is most similar to 马扎尔人 from the following options: A:Hungarians, B:Murad_I, C:North_America, D:Me_(Super_Junior-M_album). 
            A
            Choose the option that is most similar to {ent1_dic[re_rank_list[i]]} from the following options: A:{ent2_dic[re_rank_topk[i][0]]}, B:{ent2_dic[re_rank_topk[i][1]]}, C:{ent2_dic[re_rank_topk[i][2]]},
                #                         D:{ent2_dic[re_rank_topk[i][3]]}, E:{ent2_dic[re_rank_topk[i][4]]}, F:{ent2_dic[re_rank_topk[i][5]]},
                #                         G:{ent2_dic[re_rank_topk[i][6]]}, H:{ent2_dic[re_rank_topk[i][7]]}, I:{ent2_dic[re_rank_topk[i][8]]},
                #                         J:{ent2_dic[re_rank_topk[i][9]]}.

            """

        # # 候选集20个
        # prompt = f"""
        #         Choose the option that is most similar to 马扎尔人 from the following options: A:Hungarians, B:Murad_I, C:North_America, D:Me_(Super_Junior-M_album).
        #         A
        #         Choose the option that is most similar to {ent1_dic[re_rank_list[i]]} from the following options: A:{ent2_dic[re_rank_topk[i][0]]}, B:{ent2_dic[re_rank_topk[i][1]]}, C:{ent2_dic[re_rank_topk[i][2]]},
        #                         D:{ent2_dic[re_rank_topk[i][3]]}, E:{ent2_dic[re_rank_topk[i][4]]}, F:{ent2_dic[re_rank_topk[i][5]]},
        #                         G:{ent2_dic[re_rank_topk[i][6]]}, H:{ent2_dic[re_rank_topk[i][7]]}, I:{ent2_dic[re_rank_topk[i][8]]},
        #                         J:{ent2_dic[re_rank_topk[i][9]]},K:{ent2_dic[re_rank_topk[i][10]]}, L:{ent2_dic[re_rank_topk[i][11]]}, M:{ent2_dic[re_rank_topk[i][12]]},
        #         #                         N:{ent2_dic[re_rank_topk[i][13]]}, O:{ent2_dic[re_rank_topk[i][14]]}, P:{ent2_dic[re_rank_topk[i][15]]},
        #         #                         Q:{ent2_dic[re_rank_topk[i][16]]}, R:{ent2_dic[re_rank_topk[i][17]]}, S:{ent2_dic[re_rank_topk[i][18]]},
        #         #                         T:{ent2_dic[re_rank_topk[i][19]]}.
        #
        #         """

        result = get_completion(prompt)
        # print(result)
        # print(result.split('\'')[1])
        # if result.split()[0]
        # write_data_2file("re_rank_result", result.split('\'')[1])
        if len(re_rank_list) > 5000:
            write_data_2file(fold + lang + "gpt_simple_result_id" + f, result)
        else:
            write_data_2file(fold + lang + "gpt_hard_result_id" + f, result)
        time.sleep(2)


# query(s_l, ent_dic_1, ent_dic_2, simple_list, simple_topk)
query(h_l, ent_dic_1, ent_dic_2, hard_list, hard_topk)