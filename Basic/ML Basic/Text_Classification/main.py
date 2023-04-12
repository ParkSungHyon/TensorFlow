import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

dataset_dir = "/Users/b31/Documents/DataSet/aclImdb"
train_dir = os.path.join(dataset_dir, 'train')

#sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
#with open(sample_file) as f:
#    print(f.read())

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory( # 훈련 데이터셋
    '/Users/park/Documents/Python/DataSet/aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2, #전체 파일중 20%를 검증용으로 사용
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory( # 검증 데이터셋
    '/Users/park/Documents/Python/DataSet/aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2, #전체 파일중 20%를 검증용으로 사용
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory( # 테스트 데이터셋
    '/Users/park/Documents/Python/DataSet/aclImdb/test',
    batch_size=batch_size)

#for text_batch, label_batch in raw_train_ds.take(1): # 첫번째 배치를 가져옴
#    for i in range(3): # 배치 내부에서 3개의 샘플을 불러옴
#        print("Reviw", text_batch.numpy()[i])
#        print("Label", label_batch.numpy()[i])

#html 태그 삭제, 소문자로 변환
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),'')

max_features = 10000 # 이 모델에서 고려할 최대 단어 수
sequence_length = 250 # 각 샘플의 텍스트 시퀀스 최대 길이 - 일정하게 맞추기 위해

vectorize_layer = layers.TextVectorization( # 텍스트 데이터를 벡터화하여 모델의 입력으로 사용
    standardize=custom_standardization, # 텍스트 데이터 전처리 함수
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

train_text = raw_train_ds.map(lambda x, y: x)
#입력 데이터 추출하여 x에 저장 (라벨y는 제외)
vectorize_layer.adapt(train_text)
## raw_train_ds에서 입력 데이터만을 추출한 데이터셋 train_test를 이용하여
## vectorize_layer를 학습(fit)하여 텍스트 데이터를 벡터로 변환할수 있는 내부 상태를 설정
## 단어별로 숫자(int)를 자동으로 할당함 (빈도가 높을수록 낮은 숫자)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label
# 하나의 리뷰 텍스트와 해당 리뷰의 레이블을 입력으로 받아서, 텍스트를 벡터로 변환한뒤, 벡터와 레이블을 반환

text_batch, label_batch = next(iter(raw_train_ds)) # text_batch, label_batch는 tf.Tensor 객체 형태로 저장됨
first_review, first_label = text_batch[0], label_batch[0]
#print("Review", first_review)
#print("Label", raw_train_ds.class_names[first_label])
#print("Vectorized review", vectorize_text(first_review, first_label))

#print(vectorize_layer.get_vocabulary()[1])
#print(vectorize_layer.get_vocabulary()[9997])
#print(len(vectorize_layer.get_vocabulary()))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
print("type of trian_ds", type(train_ds))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])
#model.summary()

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
               optimizer='adam',
               metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
'''
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)
'''
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=['accuracy']
)
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

examples = [
    "This movie was terrible",
    "This movie was not good",
    "I'll not recommend this movie to other people",
    "It's waste of time to see this movie",
    #######################
    "this movis was very nice",
    "this movie was ok",
    "this movie had been touched my mind",
    "I'd like to see this movie again soon"
]

#print(export_model.predict(examples))
