# import torch
import numpy as np
from Param import *
# fold = "data/DBP15K/"
# lang = "zh_en/"
# k = 10
# f = str(k)
mod = "8"
tem = ""
# tem = ""

print("candidate size:", f)


def load_llama_re_7(path):
    e = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            re_ori = line.strip().replace('[', '').replace(']', '').replace('>', '').replace(' ', '').split("\',")
            re_list = [re for re in re_ori if len(re) != 0]
            e.append(re_list)
    return e


def load_llama_re_8(path):
    e = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            re_ori = line.strip().replace('[', '').replace(']', '').replace('>', '').replace(' ', '').split(":")
            re_list = [re for re in re_ori if len(re) != 0]
            e.append(re_list)
    return e





if mod == "7":
    simple_result = load_llama_re_7(fold + lang + tem + "llama" + mod + "b_simple_result_id" + f)
    hard_result = load_llama_re_7(fold + lang + tem + "llama" + mod + "b_hard_result_id" + f)
    print(7)
if mod == "8":
    simple_result = load_llama_re_8(fold + lang + tem + "llama" + mod + "b_simple_result_id" + f)
    hard_result = load_llama_re_8(fold + lang + tem + "llama" + mod + "b_hard_result_id" + f)
    print(8)



# print(hard_result)

def process_result_7(result):
    r = [x[0] if len(x) == 1 else x[1] for x in result]
    r_f = [i[1] for i in r]
    return r_f


def process_result_8(result):
    r = [x[0] if len(x) == 1 else x[1] for x in result]
    r_f = [i[0] for i in r]
    return r_f





if mod == "7":
    sr = process_result_7(simple_result)
    hr = process_result_7(hard_result)
    print(7)
if mod == "8":
    sr = process_result_8(simple_result)
    hr = process_result_8(hard_result)
    print(8)



# print(hr)

def write_re(path, data):
    with open(path, 'w', encoding='UTF-8') as f:
        for d in data:
            f.write(str(d) + "\n")
        print("Saved!")


write_re(fold + lang + "new_" + tem + "llama" + mod + "b_simple_result_id" + f, sr)
write_re(fold + lang + "new_" + tem + "llama" + mod + "b_hard_result_id" + f, hr)
