# -*- coding: utf-8 -*-
# @Time : 2020/12/23 14:19
# @Author : Jclian91
# @File : model_train.py
# @Place : Yangpu, Shanghai
import json
import codecs
import numpy as np
from keras.layers import *
from keras.models import Model
from keras_bert import AdamWarmup, calc_train_steps
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

from FGM import adversarial_training

maxlen = 128
BATCH_SIZE = 32
EPOCH = 5
config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'


token_dict = {}
with codecs.open(dict_path, 'r', 'utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)


def seq_padding(X, padding=0):
    return np.array([
        np.concatenate([x, [padding] * (maxlen - len(x))]) if len(x) < maxlen else x for x in X
    ])


class DataGenerator:

    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                first_text, second_text = d[0], d[1]
                x1, x2 = tokenizer.encode(first=first_text, second=second_text, max_len=maxlen)
                y = d[2]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = np.array(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


# 构建模型
def create_text_match_model(num_labels):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=maxlen)

    for layer in bert_model.layers:
        layer.trainable = True

    # Add Bi-LSTM layer
    bi_lstm = Bidirectional(LSTM(64, return_sequences=True))(bert_model.output)
    bi_lstm = Lambda(lambda x: x, output_shape=lambda s: s)(bi_lstm)
    print(bi_lstm.shape)
    # Applying hybrid pooling approach to bi_lstm sequence output
    avg_pool = GlobalAveragePooling1D()(bi_lstm)
    max_pool = GlobalMaxPooling1D()(bi_lstm)
    concat = concatenate([avg_pool, max_pool])
    # dropout = Dropout(0.3)(concat)
    output = Dense(num_labels, activation='softmax')(concat)
    model = Model(bert_model.input, output)
    model.summary()

    return model


if __name__ == '__main__':

    # 数据处理, 读取训练集和测试集
    print("begin data processing...")
    with open("./data/LCQMC/train.json", "r", encoding="utf-8") as f:
        train_json = [_.strip() for _ in f.readlines()]
    with open("./data/LCQMC/dev.json", "r", encoding="utf-8") as f:
        test_json = [_.strip() for _ in f.readlines()]

    labels = ["0", "1"]
    with open("label.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(zip(range(len(labels)), labels)), ensure_ascii=False, indent=2))

    train_data = []
    test_data = []
    for sample in train_json:
        sample = json.loads(sample)
        label, first_sent, second_sent = sample["label"], sample["sentence1"], sample["sentence2"]
        label_id = [0] * len(labels)
        for j, _ in enumerate(labels):
            if _ == label:
                label_id[j] = 1
        train_data.append((first_sent, second_sent, label_id))

    for data in train_data[:50]:
        print(data)

    for sample in test_json:
        sample = json.loads(sample)
        label, first_sent, second_sent = sample["label"], sample["sentence1"], sample["sentence2"]
        label_id = [0] * len(labels)
        for j, _ in enumerate(labels):
            if _ == label:
                label_id[j] = 1
        test_data.append((first_sent, second_sent, label_id))

    print("finish data processing!")

    # 模型训练
    model = create_text_match_model(len(labels))
    # 启用对抗训练FGM
    # adversarial_training(model, 'Embedding-Token', 0.5)
    # add warmup
    total_steps, warmup_steps = calc_train_steps(
        num_example=len(train_data),
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        warmup_proportion=0.1,
    )
    optimizer = AdamWarmup(total_steps, warmup_steps, lr=2e-5, min_lr=1e-7)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    train_D = DataGenerator(train_data)
    test_D = DataGenerator(test_data)

    print("begin model training...")
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=EPOCH,
        validation_data=test_D.__iter__(),
        validation_steps=len(test_D)
    )

    print("finish model training!")

    # 模型保存
    model.save('lcqmc_text_match.h5')
    print("Model saved!")

    result = model.evaluate_generator(test_D.__iter__(), steps=len(test_D))
    print("模型评估结果:", result)