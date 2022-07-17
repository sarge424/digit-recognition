import numpy as np
import pandas as pd
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import relu, linear, softmax
from tensorflow.keras.optimizers import Adam

def print_accuracy(y, yhat):
    '''
    y = [m, 1]
    yhat = [m, 1]
    '''
    
    m = len(y)
    hits = 0
    for i in range(m):
        prediction = yhat[i].argmax(axis=0)
        hits += 1 if y[i] == prediction else 0
        #print(i, y[i], prediction)
    
    acc = hits/m * 100
    print(f'{100-acc: .2f}%')



#Read The MNIST Training Set as 0 -> 1 per pixel
print('\n-->Loading data...')

mnist = pd.read_csv('data/mnist.csv').to_numpy()

#The first column holds labels
m_train = 36000

X = mnist[:m_train,1:] / 255
y = mnist[:m_train, 0]

X_test = mnist[m_train:,1:] / 255
y_test = mnist[m_train:, 0]

print(f'Training set ({len(X)}):')
print(X.shape, type(X))
print(y.shape, type(y))
print(f'\nTest set ({len(X_test)}):')
print(X_test.shape, type(X_test))
print(y_test.shape, type(y_test))

print('-->Data loaded.\n\n-->Loading model...')
model = tf.keras.models.load_model('./models')

t1 = time.time()
z = model.predict(X)
yhat = tf.nn.softmax(z).numpy()
time_train = time.time() - t1

t1 = time.time()
z_test = model.predict(X_test)
yhat_test = tf.nn.softmax(z_test).numpy()
time_test = time.time() - t1

print('\n-->Performance:')
print(f'Training set error ({time_train * 1e6 // m_train} us/x):', end='')
print_accuracy(y, yhat)
print(f'Test set error ({time_test * 1e6 // (42000 - m_train)} us/x):', end='')
print_accuracy(y_test, yhat_test)