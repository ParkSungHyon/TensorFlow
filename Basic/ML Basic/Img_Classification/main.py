import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

#print(train_images.shape)
#print(train_labels.shape)
#print(test_images.shape)
#print(test_labels.shape)

#plt.figure()
#plt.imshow(train_images[1])
#plt.colorbar()
#plt.grid(False)
#plt.show()

train_images = train_images / 255.0 #픽셀값을 0~1로 조정 -> 픽셀값의 범위가 넓을 경우 작은 가중치 업데이트로 작은 변화만 생기기 때문에 학습하는데 시간이 길어짐
test_images = test_images / 255.0
'''
plt.figure(figsize=(8,8))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)])
#model.summary()
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, verbose=1)
result = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest result:', result)
#test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
#print('\nTest loss:', test_loss)
#print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# model 객체와 tf.keras.layers.Softmax() 레이어를 연결하여 새로운 모델을 생성
predictions = probability_model.predict(test_images)
# Softmax()를 추가한 새로운 모델이 예측한 결과를 반환
# 이미지 번호에 대한 클래스별 확률값 / predictions[0]은 첫번째 이미지에 대한 예측 확률값을 포함하는 1차원 배열


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
'''
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
'''
num_rows = 8
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()