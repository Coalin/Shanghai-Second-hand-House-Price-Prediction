#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tools import normalize, num_features, HIDDEN_SIZE
from tools import oh_encode
from matplotlib import pyplot as plt

TRAINING_EXAMPLES = 2048
NUM_EPOCHS = 10000


def batch_generator(data_frame_encoded):
    """Generates data to be fed to the neural network."""
    labels = data_frame_encoded[-1]
    # data = np.delete(data_frame_encoded, -1, axis=0)
    data = data_frame_encoded[:-1]

    num_features = len(data)
    num_batches = len(data[0])
    for i in range(num_batches):
        batch_compiled = []
        for j in range(num_features):
            if type(data[j][i]) is np.ndarray:
                batch_compiled.extend(data[j][i])
            else:
                batch_compiled.extend([data[j][i]])
        yield batch_compiled, labels[i]


df_train = pd.read_csv('./data/train.csv', keep_default_na=False)
# df_train = df_train.drop(['Avg_price'], 1)
column_names = df_train.columns.values
df_train_encoded = oh_encode(df_train)
df_train_encoded_normalized = normalize(df_train_encoded)

batch_gen = batch_generator(df_train_encoded_normalized)

# create the neural network model
keep_prob = tf.placeholder(tf.float32)
prev_loss = tf.Variable(0., trainable=False)

input_layer = tf.placeholder(tf.float32, [None, num_features], name='input')
W1 = tf.Variable(tf.random_normal([num_features, HIDDEN_SIZE], stddev=.01), name='W1')
b1 = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=.01), name='b1')
h1_layer = tf.add(tf.matmul(input_layer, W1), b1)
h1_layer = tf.nn.relu(h1_layer)
h1_layer = tf.nn.dropout(h1_layer, keep_prob, name='h1')

W2 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE], stddev=.01), name='W2')
b2 = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=.01), name='b2')
h2_layer = tf.matmul(h1_layer, W2) + b2
h2_layer = tf.nn.relu(h2_layer)
h2_layer = tf.nn.dropout(h2_layer, keep_prob, name='h2')

W3 = tf.Variable(tf.random_normal([HIDDEN_SIZE, 1], stddev=.01), name='W3')
b3 = tf.Variable(tf.random_normal([1], stddev=.01), name='b3')
output_layer = tf.matmul(h2_layer, W3) + b3
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

loss = tf.squared_difference(output_layer, y)
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.01  # Choose an appropriate one.
loss = loss + reg_constant * sum(reg_losses)
loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=.85, beta2=.9).minimize(loss)

all_examples = np.array([[np.array(b), l] for b, l in batch_gen])
# split data into train and validation
idx_ = np.random.randint(0, 30920, 21644)
train_examples = all_examples[idx_]
valid_examples = all_examples[-idx_]
valid_labels = valid_examples[:, -1]
valid_labels = np.reshape(valid_labels, [-1, 1])

valid_batches = np.array(valid_examples[:, 0])
valid_len = len(valid_batches)
valid_batches = np.concatenate(valid_batches)
valid_batches = np.reshape(valid_batches, [valid_len, -1])
saver = tf.train.Saver(max_to_keep=10)
min_loss = 50000000

saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

vars = None

# get the next batch
train_loss_curve = []
valid_loss_curve = []
for i in range(NUM_EPOCHS):
    idx = np.random.randint(0, len(train_examples), TRAINING_EXAMPLES)
    train_batches = train_examples[idx]
    train_labels = np.array([train_batches[:, -1]])
    train_labels = np.reshape(train_labels, [TRAINING_EXAMPLES, 1])

    train_batches = np.array(train_batches[:, 0])
    train_batches = np.concatenate(train_batches)
    train_batches = np.reshape(train_batches, [TRAINING_EXAMPLES, -1])
    # feed the batch
    _, train_loss = sess.run([optimizer, loss], feed_dict={input_layer: train_batches, y: train_labels, keep_prob: .75})
    valid_loss = sess.run(loss, feed_dict={input_layer: valid_batches, y: valid_labels, keep_prob: 1.})
    # log results
    if i % 100 == 0:
        print('epoch: {}, train loss: {}, valid loss: {}'.format(i, train_loss, valid_loss))
    train_loss_curve.append(train_loss)
    valid_loss_curve.append(valid_loss)
    if valid_loss < min_loss:
        min_loss = valid_loss
        vars = sess.run([W1, b1, W2, b2, W3, b3])
        pickle.dump(vars, open('./saves/weights.pickle', 'wb'))
        saver.save(sess, "model.ckpt")
        print('Saved a model with loss {}'.format(valid_loss))

sess.close()

"""
plt.plot(train_loss_curve, label='$Training Loss$', color='red', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.show()

plt.plot(valid_loss_curve, label='$Testing Loss$', color='Green', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Testing Loss')
plt.title('Testing Loss')
plt.legend()
plt.show()
"""


