# -*- coding: utf-8 -*-
# @Time : 2020/12/23 15:28
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
# 模型预测脚本
import time
import json
import numpy as np
from keras.models import load_model
from keras_bert import get_custom_objects, Tokenizer

from model_train import token_dict

maxlen = 80

# 加载训练好的模型
model = load_model("arqmc_text_match.h5", custom_objects=get_custom_objects())
tokenizer = Tokenizer(token_dict)
with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())

s_time = time.time()
# 预测示例语句
text1 = "花呗收款额度限制"
text2 = "收钱码，对花呗支付的金额有限制吗"


# 利用BERT进行tokenize
X1, X2 = tokenizer.encode(first=text1, second=text2, max_len=maxlen)

# 模型预测并输出预测结果
predicted = model.predict([[X1], [X2]])
y = np.argmax(predicted[0])


print("原文: %s, %s" % (text1, text2))
print("预测标签: %s" % label_dict[str(y)])
e_time = time.time()
print("cost time:", e_time-s_time)