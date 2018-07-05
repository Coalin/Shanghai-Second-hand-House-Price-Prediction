#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelBinarizer

HIDDEN_SIZE = 200
num_features = 626
import pickle
import numpy as np


def oh_encode(data_frame, encoders_path=None):
    """Encodes categorical columns into their One-Hot representations."""
    data_encoded = []
    encoders = [] if not encoders_path else pickle.load(open(encoders_path, 'rb'))

    if encoders:
        for feature, encoder in zip(data_frame, encoders):
            data_i = list(data_frame[feature])
            if encoder is not None:
                data_i = encoder.transform(data_i)
            try:
                data_i = np.array(data_i, dtype=np.float32)
            except ValueError:
                for n, i in enumerate(data_i):
                    if i == 'NA':
                        data_i[n] = 0
                data_i = np.array(data_i, dtype=np.float32)
            data_encoded.append(data_i)
    else:
        for feature in data_frame:
            data_i = data_frame[feature]
            encoder = None
            if data_frame[feature].dtype == 'O':  # is data categorical?
                encoder = LabelBinarizer()
                encoder.fit(list(set(data_frame[feature])))
                data_i = encoder.transform(data_i)
            data_i = np.array(data_i, dtype=np.float32)
            data_encoded.append(data_i)
            encoders.append(encoder)
        pickle.dump(encoders, open('encoders.pickle', 'wb'))
    return data_encoded


def normalize(data_frame_encoded):
    """Normalize the data using log function."""
    data = data_frame_encoded
    data = [np.log(tt + 1) for tt in data]
    return data