#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:19:29 2020

@author: carolinepacheco
"""
import numpy as np
from keras.utils import np_utils


def load_text(filename, MAXLEN):
    # read text
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()

    dataX = []
    dataY = []

    chars = sorted(list(set(raw_text)))
    # convert the characters to integers
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # Summary equation dataset
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    #  convert the characters to integers using our lookup table we prepared earlier.
    seq_input = []
    seq_output = []
    for i in range(0, n_chars - MAXLEN, 1):
        seq_in = raw_text[i:i + MAXLEN]
        seq_out = raw_text[i + MAXLEN]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
        seq_input.append(seq_in)
        seq_output.append(seq_out)

    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, MAXLEN, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    return X, y, int_to_char, n_vocab, dataX, dataY, raw_text