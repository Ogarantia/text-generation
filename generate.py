#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:06:37 2020

@author: carolinepacheco
"""

import numpy as np


def generate_text(generator, dataX, n_vocab, int_to_char, max_character):
    generated = ''

    data = dataX.copy()
    start = np.random.randint(0, len(data) - 1)
    pattern = data[start]

    for i in range(max_character):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = generator.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        generated += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return generated
