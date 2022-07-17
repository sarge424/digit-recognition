import numpy as np
import pandas as pd

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

def flatten(X):
    m, n = X.shape
    for i in range(m):
        for j in range(n):
            X[i, j] = 1 if X[i, j] > 0 else 0
            
    return X

#Read The MNIST Training Set as 0 -> 1 per pixel
print('-->Loading data...')

mnist = pd.read_csv('data/mnist.csv').to_numpy()

#The first column holds labels
m_train = 36000

X = flatten(mnist[:m_train,1:] / 255)
y = mnist[:m_train, 0]

X_test = flatten(mnist[m_train:,1:] / 255)
y_test = mnist[m_train:, 0]

print(f'Training set ({len(X)}):')
print(X.shape, type(X))
print(y.shape, type(y))
print(f'\nTest set ({len(X_test)}):')
print(X_test.shape, type(X_test))
print(y_test.shape, type(y_test))

print('-->Data loaded.\n\n')
print('-->Initializing model...')

model = Sequential([
	Dense(units=100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
	Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(units=25, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(units=10, activation='linear')
])

print('-->model initialized.\n\n')
print('-->Compliling Model...')

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=1e-3)
)

print('-->model compiled.\n\n')
print('-->Training Model...')

model.fit(X, y, epochs=250)

z = model.predict(X)
yhat = tf.nn.softmax(z).numpy()

z_test = model.predict(X_test)
yhat_test = tf.nn.softmax(z_test).numpy()

print('\n\n-->Performance')
print('Training set error:', end='')
print_accuracy(y, yhat)
print('Test set error:', end='')
print_accuracy(y_test, yhat_test)

print('-->Save model? (y/n):', end='')
inp = input()
if inp[0].lower() == 'y':
    model.save('./models')