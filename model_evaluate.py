# -*- coding: utf-8 -*-
# @Time : 2020/12/23 15:28
# @Author : Jclian91
# @File : model_evaluate.py
# @Place : Yangpu, Shanghai
# 模型评估脚本
import json
import numpy as np
from keras.models import load_model
from keras_bert import get_custom_objects, Tokenizer
from sklearn.metrics import classification_report

from model_train import token_dict

maxlen = 128

# 加载训练好的模型
model = load_model("arqmc_text_match.h5", custom_objects=get_custom_objects())
tokenizer = Tokenizer(token_dict)
with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())


# 对单句话进行预测
def predict_single_text(text1, text2):
    # 利用BERT进行tokenize
    X1, X2 = tokenizer.encode(first=text1, second=text2, max_len=maxlen)

    # 模型预测并输出预测结果
    predicted = model.predict([[X1], [X2]])
    y = np.argmax(predicted[0])
    return label_dict[str(y)]


# 模型评估
def evaluate():
    with open("data/AFQMC/dev.json", "r", encoding="utf-8") as f:
        test_data = [_.strip() for _ in f.readlines()]
    true_y_list, pred_y_list = [], []
    for i, sample in enumerate(test_data):
        print("predict {} sample.".format(i))
        sample = json.loads(sample)
        true_y, first_sent, second_sent = sample["label"], sample["sentence1"], sample["sentence2"]
        pred_y = predict_single_text(first_sent, second_sent)
        print(true_y, pred_y)
        true_y_list.append(true_y)
        pred_y_list.append(pred_y)

    return classification_report(true_y_list, pred_y_list, digits=4)


output_data = evaluate()
print("model evaluate result:\n")
print(output_data)