import numpy as np
import pandas as pd
import os
import glob
import io
import warnings
import random
import tensorflow as tf
import tensorflow_hub as hub
import bert
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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


def evaluate(model1, test_data1):
    test_loss1 = 0
    test_accuracy1 = 0
    for text_batch1, label_batch1 in test_data1:
        predictions1 = model1(text_batch1)
        t_loss = loss_fn(label_batch1, predictions1)
        test_loss1 += t_loss.numpy()
        test_accuracy1 += np.sum(np.round(predictions1) == label_batch1.numpy())

    test_loss1 /= TEST_BATCHES
    test_accuracy1 = test_accuracy1 / (TEST_BATCHES * BATCH_SIZE)

    return test_loss1, test_accuracy1


tokenized_abstracts = [tokenize_abstracts(abstract) for abstract in abstracts]

abstracts_with_len = [[abstract, y[i], len(abstract)]
                 for i, abstract in enumerate(tokenized_abstracts)]

random.shuffle(abstracts_with_len)

abstracts_with_len.sort(key=lambda x: x[2])

sorted_abstracts_labels = [(abstracts_lab[0], abstracts_lab[1]) for abstracts_lab in abstracts_with_len]

processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_abstracts_labels, output_types=(tf.int32, tf.int32))

BATCH_SIZE = 32

batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

next(iter(batched_dataset))

TOTAL_BATCHES = math.ceil(len(sorted_abstracts_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

print(test_data)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(None, 227))
])


loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

EPOCHS = 10

for epoch in range(EPOCHS):
    for text_batch, label_batch in train_data:
        with tf.GradientTape() as tape:
            predictions = model(text_batch)
            loss = loss_fn(label_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Evaluate the model on the test data
    test_loss, test_accuracy = evaluate(model, test_data)

    print(f'Epoch {epoch+1}, Loss: {loss}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
