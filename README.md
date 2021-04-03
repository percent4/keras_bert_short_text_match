本项目采用Keras和Keras-bert实现短文本匹配任务。

### 维护者

- jclian91

### 数据集

#### AFQMC 蚂蚁金融语义相似度

训练集/验证集/测试集: 34334/4316/3861，每一条数据有三个属性，从前往后分别是 句子1，句子2，句子相似度标签。其中label标签，1 表示sentence1和sentence2的含义类似，0表示两个句子的含义不同。例子：{"sentence1": "双十一花呗提额在哪", "sentence2": "里可以提花呗额度", "label": "0"}

#### LCQMC语义相似度

训练集/验证集/测试集: 238766/8802/12500个，text_a和text_b分别代表两个sentence，label代表两个句子之间的相似度：1代表相似，0代表不相似

### 代码结构

```
.
├── chinese_L-12_H-768_A-12（BERT中文预训练模型）
│   ├── bert_config.json
│   ├── bert_model.ckpt.data-00000-of-00001
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   └── vocab.txt
├── data（数据集）
│   └── AFQMC
│       ├── dev.json
│       ├── test.json
│       └── train.json
├── label.json（类别词典，生成文件）
├── FGM.py（对抗训练脚本）
├── model_evaluate.py（模型评估脚本）
├── model_predict.py（模型预测脚本）
├── model_train.py（模型训练脚本）
└── requirements.txt
```

## 模型效果

#### AFQMC 蚂蚁金融语义相似度

模型参数: batch_size = 32, maxlen = 128, epoch=10

评估结果:

```
              precision    recall  f1-score   support

           0     0.7978    0.8371    0.8170      2978
           1     0.5928    0.5277    0.5583      1338

    accuracy                         0.7412      4316
   macro avg     0.6953    0.6824    0.6876      4316
weighted avg     0.7342    0.7412    0.7368      4316
```

#### LCQMC语义相似度

模型参数: batch_size = 8, maxlen = 300, epoch=3

评估结果:

```

```

### 使用对抗训练FGM前后模型效果对比

#### AFQMC 蚂蚁金融语义相似度

模型参数: batch_size = 32, maxlen = 128, epoch=10

|-|train1|train2|train3|train avg|
|---|---|---|---|---|
|使用FGM前|||||
|使用FGM后|||||

#### LCQMC语义相似度

模型参数: batch_size = , maxlen = , epoch=

|-|train1|train2|train3|train avg|
|---|---|---|---|---|
|使用FGM前|||||
|使用FGM后|||||


### 项目启动

1. 将BERT中文预训练模型chinese_L-12_H-768_A-12放在chinese_L-12_H-768_A-12文件夹下
2. 所需Python第三方模块参考requirements.txt文档
3. 自己需要分类的数据按照data/AFQMC的格式准备好
4. 调整模型参数，运行model_train.py进行模型训练
5. 运行model_evaluate.py进行模型评估
6. 运行model_predict.py对新文本进行评估