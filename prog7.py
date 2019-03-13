from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle, time
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
cifar10_dataset_folder_path = 'cifar-10-batches-py'

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """
    Load a batch of the dataset
    """
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = cPickle.load(file)

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    return (x - x.min()) / (x.max() - x.min())

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    one_hot_encode = np.array(np.zeros([len(x), 10]))
    index = 0
    for elements in x:
        one_hot_encode[index][elements] = 1
        index +=1
    return one_hot_encode

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    features = normalize(features)
    labels = one_hot_encode(labels)

    cPickle.dump((features, labels), open(filename, 'wb'))

def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 5

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # Prprocess and save a batch of training data
        _preprocess_and_save(
            normalize,
            one_hot_encode,
            features,
            labels,
            'preprocess_batch_' + str(batch_i) + '.p')

    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = cPickle.load(file)

    # load the test data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all test data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_test.p')

def load_preprocess_training(batch_id):
    """
    Load the Preprocessed Training data and return them
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = cPickle.load(open(filename, mode='rb'))

    # Return the training data
    return features, labels

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Preprocess the data with normalization and one hot encoding and save as .p file
# I have already preprocessed and saved the data, so no need to process and save again. Uncomment the following line if you want to test.
# preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

# Load the previous prepocessed and saved data
# The previous processing and loading together takes a long time. So I use cifar10.load_data(), which is faster.
"""
n_batches = 5
x_train, y_train = load_preprocess_training(1)

for batch_i in range(2, n_batches + 1):
    x, y = load_preprocess_training(batch_i)
    x_train = np.concatenate((x_train, x), axis=0)
    y_train = np.concatenate((y_train, y), axis=0)
"""

model = Sequential()

model.add(Conv2D(32, (7, 7), input_shape=(32, 32, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# The integrated load_data function is faster than my own implementation before. So I am using cifar10.load_data() to save some loading time.
(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.cifar10.load_data()
train_features = train_features/255.0
test_features = test_features/255.0
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

time_callback = TimeHistory()
history = model.fit(train_features, train_labels,
                    batch_size=64,
                    epochs=10,
                    validation_data=(test_features, test_labels),
                    verbose=1,
                    callbacks=[time_callback])

training_time = time_callback.times
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.title('Training Time')
plt.plot(range(len(training_time)), training_time, 'r', label = 'Train_Time')
plt.legend(bbox_to_anchor=(0, 1), loc=2)
plt.ylabel('time (second)')
plt.xlabel('epoch')
plt.xticks(range(10))
plt.savefig('Training Time.png')
plt.figure()
plt.title('Accuracy')
plt.plot(range(len(acc)), acc, 'b', label = 'Train_Accuracy')
plt.plot(range(len(val_acc)), val_acc, 'r', label = 'Test_Accuracy')
plt.legend(bbox_to_anchor=(0, 1), loc=2)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xticks(range(10))
plt.savefig('Training and Testing Error.png')
plt.figure()
plt.title('Loss')
plt.plot(range(len(loss)), loss, 'b', label = 'Train_Loss')
plt.plot(range(len(val_loss)), val_loss, 'r', label = 'Test_Loss')
plt.legend(bbox_to_anchor=(1, 1), loc=1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks(range(10))
plt.savefig('Training and Testing Loss.png')

# Creates a HDF5 file 'my_model.h5'
model.save('my_model.h5')
