import numpy as np
import pandas as pd
import os
import glob
import io
import warnings
import random
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras import layers
import math
import bert

warnings.filterwarnings("ignore")

dir_path = []

for dirname, _, filenames in os.walk('/Users/vinay/PycharmProjects/pythonTextProcessor/data'):
    dir_path.append(dirname)

print(dir_path)

def text_to_pandasDF(path):
    df = pd.DataFrame(columns=['Abstracts', 'class'])
    txt = []
    label = []

    for dirpath in path:
        text_files_path = sorted(glob.glob(os.path.join(dirpath, '*.txt')))
        for text_path in text_files_path:
            with io.open(text_path, 'r', encoding='utf-8', errors='ignore') as txt_file:
                txt.append(txt_file.read())
                label.append(dirpath.split('/')[-1])

    df['Abstracts'] = txt
    df['class'] = label
    txt, label = [], []

    return df

df = text_to_pandasDF(dir_path[1:])

abstracts = []
sentences = list(df['Abstracts'])
for sen in sentences:
    abstracts.append(sen)

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

def tokenize_abstracts(text_abstracts):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_abstracts))

y = df['class']
y = np.array(list(map(lambda x: 1 if x=="Biochem" else 0, y)))

print(f"Abstract sample:\n {abstracts[10]}")
print(f"Abstract Class: {y[10]}")

tokenized_abstracts = [tokenize_abstracts(abstract) for abstract in abstracts]

abstracts_with_len = [[abstract, y[i], len(abstract)]
                 for i, abstract in enumerate(tokenized_abstracts)]

random.shuffle(abstracts_with_len)

abstracts_with_len.sort(key=lambda x: x[2])

sorted_abstracts_labels = [(abstracts_lab[0], abstracts_lab[1]) for abstracts_lab in abstracts_with_len]

processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_abstracts_labels, output_types=(tf.int32, tf.int32))

BATCH_SIZE = 32

batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

next(iter(batched_dataset))

TOTAL_BATCHES = math.ceil(len(sorted_abstracts_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

class TextClassificationModel(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 name="text_model"):

        super(TextClassificationModel, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2
DROPOUT_RATE = 0.2
NB_EPOCHS = 5

text_model = TextClassificationModel(
                 vocabulary_size=VOCAB_LENGTH,
                 embedding_dimensions=EMB_DIM,
                 cnn_filters=CNN_FILTERS,
                 dnn_units=DNN_UNITS,
                 model_output_classes=OUTPUT_CLASSES,
                 dropout_rate=DROPOUT_RATE)

if OUTPUT_CLASSES == 2:
    text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
else:
    text_model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])

text_model.fit(train_data, epochs=NB_EPOCHS)

results = text_model.evaluate(test_data)
print(f"Test evaluation results: {results}")

