from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras

# import helper libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# import fashion data from keras dataset
'''
* train_images: (60000 x 28 x 28), 0~255 픽셀 데이터.
* train_labels: (60000), 0~9 레이블 데이터.
* test_images: (10000 x 28 x 28), 0~355 픽셀 데이터.
* test_labels: (10000), 0~9 레이블 데이터
'''
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_images(imgs, num_shows):
    num_cols = 5
    num_rows = int(np.ceil(num_shows / num_cols))
    plt.figure(figsize=(10,10))
    for i in range(num_shows):
        plt.subplot(num_rows, num_cols, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i], cmap=cm.get_cmap('binary'))
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

# data whitening
train_images = train_images / 255.0
test_images = test_images / 255.0

#plot_images(train_images, 25)

# construct model
## construct layers
model = keras.Sequential([
    # Flatten - input_shape을 1차원 배열로 flatten해주는 레이어
    keras.layers.Flatten(input_shape=(28,28)),
    # Dense - relu activator 밀집층. 128개의 노드(뉴런)로 구성됨
    keras.layers.Dense(128, activation=tf.nn.relu),
    # Dense - softmax activator 밀집층. 10개의 노드로 구성되고 모든 노드의 총합은 1
    # (softmax multiclass classification 특징)
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

## compile model
'''
본격적인 훈련에 앞서 필요한 설정들을 컴파일단계에서 추가함
'''
model.compile(
    # 데이터와 loss로부터 모델의 업데이트 방법을 결정. gradient decent같은걸 지정할 수 있을 듯
    optimizer='adam',
    # 훈련하는 동안 모델의 오차. 이 오차가 작아지는 방향으로 훈련됨
    loss='sparse_categorical_crossentropy',
    # 훈련 및 테스트 단계를 모니터링하기 위해 사용됨.
    metrics=['accuracy']
)

# train model
model.fit(train_images, train_labels, epochs=5)

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test accuracy: ', test_acc)

# create predictions and plot
'''
실제 대상의 레이블을 예측하기. 예제에서는 test_images에 대해 수행함
'''
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=cm.get_cmap('binary'))

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

## predict for one image
img = test_images[0]
img = (np.expand_dims(img, 0)) # 이미지 하나만 사용해도 tf에 넣을때는 배치 형식으로 넣어줘야함. (1 x 28 x 28)

prediction_single = model.predict(img)
plot_value_array(0, prediction_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()