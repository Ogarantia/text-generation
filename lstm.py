#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:13:27 2020

@author: carolinepacheco
"""

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


def build_generator(X, y, UNIT, DROPOUT, OPTIMIZER):
    print('Building generator model...')
    generator = Sequential()
    generator.add(LSTM(UNIT, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    generator.add(Dropout(DROPOUT))
    generator.add(LSTM(UNIT))
    generator.add(Dropout(DROPOUT))
    generator.add(Dense(y.shape[1], activation='softmax'))
    generator.summary()

    return generator
