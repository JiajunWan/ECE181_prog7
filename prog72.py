from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *

(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.cifar10.load_data()
train_features = train_features/255.0
test_features = test_features/255.0
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# Returns a compiled model identical to the previous one
model = load_model('my_model.h5')

scores = model.evaluate(test_features, test_labels, verbose=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
