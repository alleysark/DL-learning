from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# download IMDB(Internet Movie Database) dataset
''' 영화 리뷰 텍스트 데이터 임포트 (상위 1만단어만 사용). '''
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# IMDB data at a glance
print('훈련 샘플: {}, 레이블: {}'.format(len(train_data), len(train_labels)))

'''
IMDB 데이터는 영화 리뷰 텍스트에 대한 전처리된 정수 배열. 각 정수는 단어에 매치됨.
'''
print(train_data[0])

'''
그리고 각 리뷰 텍스트는 길이도 다르다. 신경망은 입력 데이터 크기가 같아야하므로 이를 맞춰줘야함.
'''
print(len(train_data[0]), len(train_data[1]))

## convert word integers to matched string
word_index = imdb.get_word_index()

word_index = { k:(v+3) for k,v in word_index.items() }
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2  # unknown
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(w, '?') for w in text])

print(decode_review(train_data[0]))

# prepare data
'''
신경망은 동일한 크기의 입력을 사용해야함. 따라서 가변 길이 텍스트 데이터를 동일 크기의 텐서로 변환해야 한다.
1. one-hot encoding: num_words 크기의 벡터로, 등장하는 단어는 1 그렇지 않으면 0인 거대한 벡터.
2. padding: max_length 가 되도록 나머지 텍스트에 패딩을 추가하는 방법
이 외에도 자연어 처리에서 텍스트를 변환하는 방법은 다양하지만, 예제에서는 2 번 방법을 사용
'''
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=256)

# construct model
'''
신경망 층(layer) 구조에서 고려할 것:
* 모델에서 얼마나 많은 층을 사용할 것인가?
* 각 층에서 얼마나 많은 hidden unit을 사용할 것인가?
'''
vocab_size = 10000

model = keras.Sequential()
''' 정수로 인코딩된 단어를 입력받고 각 단어 인덱스에 해당하는 임베딩 벡터를 찾음. 이 벡터는 모델이 훈련되면서 학습됨. 
이 벡터는 출력 배열에 새로운 찬원으로 추가됨. 최종 차원은 (batch, sequence, embedding) '''
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
''' sequence 차원에 대해 평균을 계산하여 각 샘플에 대해 고정된 길이의 출력 벡터를 반환. 길이가 다른 입력을 다루는 가장 간단한 방법 '''
model.add(keras.layers.GlobalAveragePooling1D())
''' 위 고정 길이 출력 벡터는 16개의 hidden unit을 가진 fully-connected 층(Dense)을 거치게 됨 '''
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
''' 마지막 층은 하나의 출력 노드를 가지는 fully-connected 층. sigmoid로 [0,1] 실수를 출력함. a.k.a 신뢰도 '''
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# loss and optimizer
'''
0 or 1 classification 문제에서, 확률을 다루는 데에는 binary crossentropy가 적합. mean squared error를 선택할 수도 있지만, 이는 regression 문제에 더 적합함
'''
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='binary_crossentropy',
    metrics=['acc'])

# create validation set
num_vals = 10000
x_val = train_data[:num_vals]
partial_x_train = train_data[num_vals:]

y_val = train_labels[:num_vals]
partial_y_train = train_labels[num_vals:]

# train model
''' train model as a 512 sample mini-batch for 40 epoch '''
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1)

# accuracy and loss graph
history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(epochs, loss, 'bo', label='Training loss')
ax1.plot(epochs, val_loss, 'b', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, acc, 'bo', label='Training acc')
ax2.plot(epochs, val_acc, 'b', label='Validation acc')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()