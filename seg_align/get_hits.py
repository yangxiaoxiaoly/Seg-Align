
import torch
import numpy as np
from Param import *
# fold = "data/DBP15K/"
# lang = "zh_en/"
# k = 10
# f = str(k)
print("candidate size:", f)
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
pair_num = len(ent1)

ent1_dic = ent_to_dic(ent1)
ent2_dic = ent_to_dic(ent2)



top_1 = np.load(fold + lang + 'top1.npy')
top_10 = np.load(fold + lang + 'top10.npy')

re_rank_list = np.load(fold + lang + 'hard_list_' + f + '.npy')
re_rank_top10 = np.load(fold + lang + 'hard_top' + f + '.npy')


def load_re_id(path):
    e = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            e.append(line.strip())
    return e

re_rank_result = load_re_id(fold + lang + "gpt_hard_result_id" + f)


id_dic = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9}

re_rank_id = []
c = 0
for i in range(len(re_rank_list)):
    if re_rank_result[i] in id_dic:
        id_r = re_rank_top10[i][id_dic[re_rank_result[i]]]
        re_rank_id.append(id_r)
    else:
        re_rank_id.append(0)

# print(c)
# print(re_rank_id)
# print(len(re_rank_list))
rest = []
for i in range(10500):
    if i not in re_rank_list:
        rest.append(i)
# print(len(rest))


top_1[re_rank_list] = torch.tensor(re_rank_id).view(-1, 1)
# print(re_rank_id)
# print(top_1[:100])

H1 = (torch.tensor(top_1) == torch.arange(pair_num).view(-1, 1)).sum().item()/pair_num
print(lang,H1)

