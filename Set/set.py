# https://hmkim312.github.io/posts/MNIST_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A1%9C_%ED%95%B4%EB%B3%B4%EB%8A%94_%EB%94%A5%EB%9F%AC%EB%8B%9D_(Deep_Learning)/
# https://velog.io/@skarb4788/%EB%94%A5-%EB%9F%AC%EB%8B%9D-MNIST-%EB%8D%B0%EC%9D%B4%ED%84%B0
# https://sjpyo.tistory.com/58
# https://www.tensorflow.org/guide/keras/train_and_evaluate?hl=ko

import tensorflow as tf
#print("tf v.:", tf.__version__)

mnist = tf.keras.datasets.mnist
# MNIST : 원본데이터는 캐글에서 다운로드 가능https://www.kaggle.com/oddrationale/mnist-in-csv
# MNIST 데이터는 TensorFlow에 내장되어 있기 때문에 위와 같이 사용 가능

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# mnist.load_data() Returns -> Tuple of NumPy arrays: (x_train, y_train), (x_test, y_test)
# x_train -> 데이터 순서(1/60000)와 해당되는 순서의 28*28 픽셀 값
# x_train.shape -> (60000, 28, 28) -> (몇번째데이터, 픽셀x축, 픽셀y축)
# y_train.shape -> (60000,) -> 몇번째데이터(레이블)
# 출력 예시 print(x_train[200, 15, 12]) -> output:168
#         print(x_train[200]) -> output:200번째 데이터의 28*28 배열이 출력
#         print(y_train[200,]) -> output:200번째 데이터의 레이블
x_train, x_test = x_train / 255.0, x_test / 255.0
# 할당된 데이터를 정수에서 부동 소수점 숫자로 변환


# 모델 생성
model = tf.keras.models.Sequential([ #Sequential 모델은 하나의 입력 텐서와 하나의 출력 텐서가 있는 일반 레이어 스택에 적합
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2), # 이전 레이어 노드의 20%를 제거 (과적합 방지)
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()
# 모델의 구성과 summary 출력값이 동일한지 확인

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', #다중분류에 사용되는 손실함수
              metrics=['accuracy'])

'''
predictions = model(x_train[:1]).numpy()
print(x_train[:1])
print(predictions)
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
'''
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

'''
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
probability_model(x_test[:5])
'''

import numpy as np
predicted_result = model.predict(x_test)
predicted_labels = np.argmax(predicted_result, axis = 1)
wrong_result = []
for n in range(0, len(y_test)):
    if predicted_labels[n] != y_test[n]:
        wrong_result.append(n)
print(len(wrong_result))

import random
samples = random.choices(population=wrong_result, k = 16)
print(samples)

import matplotlib.pyplot as plt
plt.figure(figsize=(14,12))
for idx, n in enumerate(samples):
    plt.subplot(4, 4, idx + 1)
    plt.imshow(x_test[n].reshape(28, 28), cmap = 'Greys', interpolation = 'nearest')
    plt.title('Label : ' + str(y_test[n]) + 'Predict : ' + str(predicted_labels[n]))
    plt.axis('off')
    
plt.show()