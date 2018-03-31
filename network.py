'''
File: \network.py
Project: NumberRecongization
Created Date: Monday March 26th 2018
Author: Huisama
-----
Last Modified: Saturday March 31st 2018 11:38:46 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import tensorflow as tf
from resource import DataSet, STD_HEIGHT, STD_WIDTH
keras = tf.contrib.keras
import numpy as np
import matplotlib.pyplot as plt

import os
import time

import shutil

Sequential = keras.models.Sequential

load_model = keras.models.load_model

Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D

SGD = keras.optimizers.SGD
Adagrad = keras.optimizers.Adagrad

BatchNormalization = keras.layers.BatchNormalization

TruncatedNormal = keras.initializers.TruncatedNormal

DATA_DIR = './Pic'
MODEL_DIR = './Model'
LOG_FILE = './training.log'

class NetWork(object):
    def __init__(self):
        self.dataset = DataSet(DATA_DIR, 8)

    '''
        Display every output of a layer
    '''
    def display_layers_shape(self, model, flatten = False):
        if flatten:
            _, s = model.output_shape
            print("Output shape %s" % s)
        else:
            _, s1, s2, s3 = model.output_shape
            print("Output shape: (%s, %s, %s), total %s" % (s1, s2, s3, s1 * s2 * s3))

    '''
        Build network
    '''
    def build_network(self):
        model = Sequential()

        model.add(Conv2D(64, (7, 7), activation='relu',
            input_shape = (STD_HEIGHT, STD_WIDTH, 6), strides = (1, 1),
            padding = 'same'))
        model.add(BatchNormalization())
        self.display_layers_shape(model)

        model.add(Conv2D(128, (5, 5), activation='relu', strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization())
        self.display_layers_shape(model)
        
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), activation='relu', strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization())
        self.display_layers_shape(model)

        model.add(Conv2D(512, (2, 2), activation='relu', strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization())
        self.display_layers_shape(model)

        model.add(Conv2D(512, (2, 2), activation='relu', strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization())
        self.display_layers_shape(model)

        model.add(Conv2D(1024, (2, 2), activation='relu', strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization())
        self.display_layers_shape(model)

        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        self.display_layers_shape(model, flatten=True)

        model.add(Dense(2048, activation='relu'))
        model.add(BatchNormalization())

        self.display_layers_shape(model, flatten=True)

        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())

        self.display_layers_shape(model, flatten=True)

        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())

        self.display_layers_shape(model, flatten=True)

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
                
        self.display_layers_shape(model, flatten=True)


        model.add(Dense(1, activation='sigmoid'))

        self.display_layers_shape(model, flatten=True)

        # ada = Adagrad()
        # model.compile(loss='binary_crossentropy', optimizer=ada)
        model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

        self.model = model

    '''
        Do training
    '''
    def train(self, episode, batch_size):
        if os.path.exists(MODEL_DIR) == False:
            os.mkdir(MODEL_DIR)
        else:
            shutil.rmtree(MODEL_DIR)
            os.mkdir(MODEL_DIR)

        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

        self.dataset.load_dataset()
        self.dataset.generate_dataset()

        for i in range(episode):
            print('Turn %d :' % (i + 1))
            data, labels = self.dataset.next_batch(batch_size)
            self.model.fit(data, labels, batch_size = 8)
            if (i + 1) % 20 == 0 and i != 0:
                score = self.validate()
                # Save model
                model_file = '%s/%s.ckpt' % (MODEL_DIR, (i + 1) / 20)
                self.model.save(model_file)
                print('\n')
                print('Model saved as %s\n' % model_file)

                with open('training.log', mode = 'a') as file:
                    file.write('Model %s, loss: %s, acc: %s\n' % ((i + 1) / 20, score[0], score[1]))
            if (i + 1) % 30 == 0 and i != 0:
                self.test()

    '''
        Do validation
    '''
    def validate(self):
        # data, labels = self.dataset.get_train_set()
        # score = self.model.evaluate(data, labels, batch_size = 8)
        # print("training result: ", score)
        data, labels = self.dataset.get_validation_set()
        score = self.model.evaluate(data, labels, batch_size = 8)
        print("validation result: ", score)
        return score

    '''
        Do test
    '''
    def test(self):
        data, labels = self.dataset.get_test_set()
        score = self.model.predict(data)
        score = np.array([0 if i < 0.5 else 1 for i in score])

        result = (score - labels).tolist().count(0) / len(score)
        print("result:", score)
        print("labels:", labels)
        print(result)

    '''
        Load model
    '''
    def load_model(self, model_path):
        self.model = load_model(model_path)

    '''
        Predict returning Ture or False
    '''
    def predict(self, data):
        result = self.model.predict(data)
        
        if result[0] < 0.5:
            return False
        else:
            return True
