#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:06:37 2020

@author: carolinepacheco
"""

from dataset import load_text
from generate import generate_text
from lstm import build_generator
from train import train_equation

unit = 700
dropout = 0.004
optomizer = "Adam"

batch_size = 40
epochs = 2030

# valid equation
filename = "dataset/alchemist.txt"
X, y, int_to_char, n_vocab, dataX, dataY, raw_text = load_text(filename, 100)

generator = build_generator(X, y, unit, dropout, optomizer)
generator.compile(loss='categorical_crossentropy', optimizer=optomizer)

# train model
model = train_equation(generator, epochs, batch_size, X, y)

# generate text
generated = generate_text(model, dataX, n_vocab, int_to_char, 400)
print(generated)

