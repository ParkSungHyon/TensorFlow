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
    '/Users/b31/Documents/DataSet/aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2, #전체 파일중 20%를 검증용으로 사용
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory( # 검증 데이터셋
    '/Users/b31/Documents/DataSet/aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2, #전체 파일중 20%를 검증용으로 사용
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory( # 테스트 데이터셋
    '/Users/b31/Documents/DataSet/aclImdb/test',
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

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_token=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


