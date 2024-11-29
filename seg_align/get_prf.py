import numpy as np
from Param import *
# fold = "data/DBP15K/"
# lang = "zh_en/"
# k = 10
# f = str(k)
print("candidate size:", f)
mod = "8"
tem = "tem0_"


model = "new_"+ tem +"llama" + mod + "b_"
print("model:", model)


# load entities

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
top_10 = np.load(fold + lang + 'top'+ f + '.npy')

# re_rank_list = np.load(fold + lang + 're_rank_list'+ f + '.npy')
# re_rank_top10 = np.load(fold + lang + 're_rank_top10.npy')

hard_list = np.load(fold + lang + 'hard_list_'+ f + '.npy')
hard_top10 = np.load(fold + lang + 'hard_top'+ f + '.npy')

simple_list = np.load(fold + lang + 'simple_list_'+ f + '.npy')
simple_top10 = np.load(fold + lang + 'simple_top'+ f + '.npy')

def load_re_id(path):
    e = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            e.append(line.strip())
    return e

simple_result = load_re_id(fold + lang + model + "simple_result_id" + f)
hard_result = load_re_id(fold + lang + model + "hard_result_id" + f)


print(lang)
h_l = len(hard_list)
s_l = len(simple_list)

print(lang)
print("number of hard sample:", h_l)
print("number of simple sample:", s_l)



# print(hard_result)
def result_to_id(re_rank_list, re_rank_result, re_rank_top10):
    id_dic = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9}
    # id_dic = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "There": 100}
    # id_dic = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10,"L": 11,"M": 12,"N": 13,"O": 14,"P": 15,"Q": 16,"R": 17,"S": 18,"T": 19,"There": 100}
    re_rank_id = []
    c = 0
    for i in range(len(re_rank_list)):
        if re_rank_result[i] not in id_dic.keys():
            re_rank_id.append(None)
            c += 1
        else:
            # if id_dic[re_rank_result[i]] == 100:
            #     re_rank_id.append(None)
            #     c += 1
            # else:
            id = re_rank_top10[i][id_dic[re_rank_result[i]]]
            re_rank_id.append(id)
    # re_rank_id = []
    # c = 0
    # for i in range(len(re_rank_list)):
    #     if re_rank_result[i] == 100:
    #         re_rank_id.append(None)
    #         c += 1
    #     else:
    #         # if id_dic[re_rank_result[i]] == 100:
    #         #     re_rank_id.append(None)
    #         #     c += 1
    #         # else:
    #         id = re_rank_top10[i][re_rank_result[i]]
    #         re_rank_id.append(id)
    return re_rank_id, c

    # if ent2_dic[id] == ent2_dic[re_rank_list[i]]:
    #     c += 1
# print(c)


# rest = []
# for i in range(10500):
#     if i not in re_rank_list:
#         rest.append(i)
# # print(len(rest))

simple_rerank_id , s_c = result_to_id(simple_list, simple_result, simple_top10)
hard_rerank_id , h_c = result_to_id(hard_list, hard_result, hard_top10)


# LLM calculate P,R,F1
#计算大模型在十个候选实体中的PRF1

def caculate_prf_llm(list, rerank_id, c):
    t_l = 0
    for i in range(len(list)):
        if list[i] == rerank_id[i]:
            t_l += 1

    P = t_l/(len(list) - c)
    R = t_l/(len(list))
    F1 = 2 * ((P * R)/(P + R))

    return P, R, F1

P_h_llm, R_h_llm, F1_h_llm = caculate_prf_llm(hard_list,hard_rerank_id, h_c)
P_s_llm, R_s_llm, F1_s_llm = caculate_prf_llm(simple_list,simple_rerank_id, s_c)


print("LLM_hard")
print("P:",P_h_llm,"R:",R_h_llm,"F1:",F1_h_llm)

print("LLM_simple")
print("P:",P_s_llm,"R:",R_s_llm,"F1:",F1_s_llm)


# SLM calculate P,R,F1
def caculate_prf_slm(list, top1):
    ori_tl = 0
    for i in range(len(top1)):
        if i in list:
            if top_1[i][0] == i:
                ori_tl += 1

    P_s = ori_tl/len(list)
    R_s = ori_tl/len(list)
    F1_s = 2 * P_s * R_s / (P_s+R_s)

    return P_s,R_s,F1_s

P_s_slm, R_s_slm, F1_s_slm = caculate_prf_slm(simple_list, top_1)
P_h_slm, R_h_slm, F1_h_slm = caculate_prf_slm(hard_list, top_1)

print("SLM_hard")
print("P_s:",P_h_slm,"R_s:",R_h_slm,"F1_s:",F1_h_slm)

print("SLM_simple")
print("P_s:",P_s_slm,"R_s:",R_s_slm,"F1_s:",F1_s_slm)


