import torch
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from Param import *

print("candidate size:", f)
#load test data
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
        ent_dic[i] = e
        i += 1
    return ent_dic

ent1, ent2 = load_ent(fold + lang + "test_links")

val_ent1, val_ent2 = load_ent(fold + lang + "valid_links")

ent1_dic = ent_to_dic(ent1)
ent2_dic = ent_to_dic(ent2)

val_ent1_dic = ent_to_dic(val_ent1)
val_ent2_dic = ent_to_dic(val_ent2)

pair_num = len(ent1)


emb1 = torch.load(fold + lang + "test_emb_1.pt")
emb2 = torch.load(fold + lang + "test_emb_2.pt")

val_emb1 = torch.load(fold + lang + "valid_emb_1.pt")
val_emb2 = torch.load(fold + lang + "valid_emb_2.pt")

S = torch.cdist(emb1, emb2, p=1)

val_S = torch.cdist(val_emb1, val_emb2, p=1)

top1 = S.topk(1, largest=False)[1]
top_k = S.topk(k, largest=False)[1]

# H1 = (top1 == torch.arange(pair_num).view(-1, 1)).sum().item()/pair_num
#
# print(lang)
# print("Hits@1")
# print(H1)


np.save(fold + lang + 'top1.npy', np.array(top1))
np.save(fold + lang + 'top'+ f + '.npy', np.array(top_k))

val_top1 = val_S.topk(1, largest=False)[1]
val_top_k = val_S.topk(k, largest=False)[1]


distance = S.topk(k, largest=False)[0]
val_distance = val_S.topk(k, largest=False)[0]

dis_array = distance.numpy()
val_dis_array = val_distance.numpy()



dis_dic = {}
for i in range(len(val_dis_array)):
    dis_dic[i] = val_dis_array[i]


cou = 0
label = []
for i in top1:
    if i.item() != cou:
        label.append(0)
        # error_list.append((cou, i.item()))
    else:
        # correct_list.append((cou, i.item()))
        label.append(1)
    cou += 1



val_label = []
val_cou = 0
for i in val_top1:
    if i.item() != val_cou:
        val_label.append(0)
        # error_list.append((cou, i.item()))
    else:
        # correct_list.append((cou, i.item()))
        val_label.append(1)
    val_cou += 1
all_data = {}
train_data = {}
for i, d in zip(val_label, list(dis_dic.keys())):
    # print(dis)
    all_data[d] = i

f_n = 0
for key in all_data:
    if all_data[key] == 0:
        if dis_dic[key].tolist()[1] - dis_dic[key].tolist()[0] < alpha:
            # print(dis_dic[key].tolist())
            train_data[key] = 0
            f_n += 1
        # if f_n == 150:
        #     break

print("the negative samples is:")
print(len(train_data))
t_n = 0
for key in all_data:
    if all_data[key] == 1:
        if dis_dic[key].tolist()[1] - dis_dic[key].tolist()[0] > beta:
            train_data[key] = 1
            t_n += 1
    if t_n == TN:
        break


X_train = []
Y_train = []
for k in train_data:
    X_train.append(dis_dic[k].tolist())
    Y_train.append(train_data[k])



X_train, X_test, y_train, y_test = np.array(X_train), dis_array, np.array(Y_train), label

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVM模型
svm_model = SVC(kernel='linear', C=1.0)  # 使用线性核，C是正则化参数

# 训练模型
svm_model.fit(X_train, y_train)
# 预测
y_pred = svm_model.predict(X_test)

# print(y_pred)

re_rank_list = [index for index, value in enumerate(y_pred) if value == 0]
print("number of hard sample：")
print(len(re_rank_list))
#
# print(top_k[re_rank_list])

rest = []
for i in range(10500):
    if i not in re_rank_list:
        rest.append(i)


print("number of simple sample：")
print(len(rest))


np.save(fold + lang + 'simple_list_'+ f + '.npy', np.array(rest))
np.save(fold + lang + 'hard_list_'+ f + '.npy', np.array(re_rank_list))

np.save(fold + lang + 'hard_top'+ f + '.npy', np.array(top_k[re_rank_list]))
np.save(fold + lang + 'simple_top'+ f + '.npy', np.array(top_k[rest]))



# print(re_rank_list)

# print(top_k[re_rank_list])
print("Saved!")

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)
