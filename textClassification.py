from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np

# download the dataset
imdb = keras.datasets.imdb

# get the 10k most frequent words and put into 
# train and test data sets
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# look at the data
#print("Training entries: {}, labels: {}".format(len(train_data), len(test_data)))

# the words of the review are assigned a number. 
#print(train_data[0])

# decode the review
# dictionary mapping words to integers
word_index = imdb.get_word_index()

# First indicies are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# print the first review
#print(decode_review(train_data[0]))

# prepare the data
# reviews must be same length
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

# build the model
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# print model summary
#model.summary()

# compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# create a validation test set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# accuracy on train_set = 97%
# accuracy on test_test = 87%       
results = model.evaluate(test_data, test_labels)
print(results)

