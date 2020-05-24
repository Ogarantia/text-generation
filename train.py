#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:19:29 2020

@author: carolinepacheco
"""

import numpy as np


def train_equation(generator, epochs, batch_size, X, y):
    for epoch in range(epochs):
        # Select a random batch of images
        idx = np.random.randint(0, X.shape[0], batch_size)
        equation = X[idx]
        label = y[idx]

        #  Train Generator
        g_loss = generator.train_on_batch(equation, label)
        print("%d[G loss: %f]" % (epoch, g_loss))

    file_path = 'model/weights-improvement.hdf5'
    generator.save_weights(file_path)

    return generator
