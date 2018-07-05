#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tools import oh_encode, normalize, num_features


def batch_generator(data):
    """Generates data to be fed to the neural network."""

    num_features = len(data)
    num_batches = len(data[0])
    for i in range(num_batches):
        batch_compiled = []
        for j in range(num_features):
            if type(data[j][i]) is np.ndarray:
                batch_compiled.extend(data[j][i])
            else:
                batch_compiled.extend([data[j][i]])
        yield batch_compiled


keep_prob = 1.

df_train = pd.read_csv('./data/new_test.csv', keep_default_na=False)
ids = df_train['Id']
df_train = df_train.drop(['Id'], 1)
# df_train = df_train.drop(['Avg_price'], 1)
column_names = df_train.columns.values
df_train_encoded = oh_encode(df_train, encoders_path='encoders.pickle')
df_train_encoded_normalized = normalize(df_train_encoded)

vars = pickle.load(open('./saves/weights.pickle', 'rb'))

input_layer = tf.placeholder(tf.float32, [None, num_features])
W1 = tf.constant(vars[0])
b1 = tf.constant(vars[1])
h1_layer = tf.add(tf.matmul(input_layer, W1), b1)
h1_layer = tf.nn.relu(h1_layer)
h1_layer = tf.nn.dropout(h1_layer, keep_prob)

W2 = tf.constant(vars[2])
b2 = tf.constant(vars[3])
h2_layer = tf.matmul(h1_layer, W2) + b2
h2_layer = tf.nn.relu(h2_layer)
h2_layer = tf.nn.dropout(h2_layer, keep_prob)

W3 = tf.constant(vars[4])
b3 = tf.constant(vars[5])
output_layer = tf.matmul(h2_layer, W3) + b3

gen = batch_generator(df_train_encoded_normalized)

all_batches = [b for b in gen]

prices = None

with tf.Session() as sess:
    prices = sess.run([output_layer], feed_dict={input_layer: all_batches})
    prices = np.array(prices).flatten()
    prices = np.exp(prices) + 1.
    print(prices)
sess.close()

ids_prices = [[id, sp] for id, sp in zip(ids, prices)]
df_prices = pd.DataFrame(ids_prices)
df_prices.to_csv('./data/result.csv', index=False, header=['Id', 'Predicted Price'])

"""
result = pd.read_csv('./data/result.csv', keep_default_na=False)
plt.scatter(range(9271),result['Predicted Price'], label='$Predicted Price$', color='Red', linewidth=1)
plt.scatter(range(9271),result['Actual Price'], label='$Acutual Price$', color='Green', linewidth=0.5)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Predicted VS Actual Price')
plt.legend()
plt.show()
"""