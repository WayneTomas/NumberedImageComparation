'''
File: \resource.py
Project: NumberRecongization
Created Date: Monday March 26th 2018
Author: Huisama
-----
Last Modified: Saturday March 31st 2018 11:08:21 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import os
import scipy.misc as scm
import random
import numpy as np

import PIL

# STD_WIDTH = 667
# STD_HEIGHT = 83
STD_WIDTH = 252
STD_HEIGHT = 40

import matplotlib.pyplot as plt

'''
    This class stands for dataset and provides data processing oparations
'''
class DataSet(object):
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_set_ratio = 0.8
        self.validate_set_ratio = 0.1

    '''
        Get mean width and height of dataset
    '''
    def get_data_mean_size(self):
        full_width, full_height = 0, 0
        count = 0

        def dummy(self, dir, file):
            nonlocal full_width, full_height, count
            filename = os.path.splitext(file)
            if filename[1] == '.png':
                fullfile = os.path.join(self.data_dir, dir, file)
                width, height = self.get_size(fullfile)
                full_width += width
                full_height += height
                print("%s, %s" % (width, height))
                count += 1

        self.lookup_dataset_dir(dummy)
        return full_width / count, full_height / count

    '''
        Get width and height of a single image
    '''
    def get_size(self, image_file_path):
        img = scm.imread(image_file_path)
        return img.shape[1], img.shape[0]

    '''
        Load dataset
    '''
    def load_dataset(self):
        self.neg_data = []
        self.pos_data = []

        self.poscount = 0
        self.negcount = 0

        def dummy(self, dir, file):
            if file == 'dataset.txt':
                # open and read in
                with open(os.path.join(self.data_dir, dir, file)) as file:
                    for line in file:
                        newline = line.strip()
                        splittext = newline.split('\t')
                            
                        if int(splittext[2]) == 1:
                            self.pos_data.append((
                                os.path.join(self.data_dir, dir, splittext[0]),
                                os.path.join(self.data_dir, dir, splittext[1]),
                                int(splittext[2])))
                            self.poscount += 1
                        else:
                            self.neg_data.append((
                                os.path.join(self.data_dir, dir, splittext[0]),
                                os.path.join(self.data_dir, dir, splittext[1]),
                                int(splittext[2])))
                            self.negcount += 1

        self.lookup_dataset_dir(dummy)

        # print("negcount: %d, poscount: %d" % (self.negcount, self.poscount))

        return True

    '''
        Check if image has 4 channel
    '''
    def check_image_channels(self):
        def dummy(self, dir, file):
            filename = os.path.splitext(file)
            if filename[1] == '.png':
                fullfile = os.path.join(self.data_dir, dir, file)
                img = scm.imread(fullfile)
                if img.shape[2] != 3:
                    print("Wrong image: %d", fullfile)


        self.lookup_dataset_dir(dummy)

    '''
        Generate dataset after loading dataset
    '''
    def generate_dataset(self):
        random.shuffle(self.neg_data)
        random.shuffle(self.pos_data)
        # total = len(self.data)
        
        pos_total = len(self.pos_data)
        pos_train_size = int(pos_total * self.train_set_ratio)
        pos_validate_size = int(pos_total * self.validate_set_ratio)
        # pos_test_size = pos_total - pos_train_size - pos_validate_size

        neg_total = len(self.neg_data)
        neg_train_size = int(neg_total * self.train_set_ratio)
        neg_validate_size = int(neg_total * self.validate_set_ratio)
        # neg_test_size = neg_total - neg_train_size - neg_validate_size

        self.batch_index = 0

        self.pos_train_set = self.pos_data[0 : pos_train_size]
        pos_validation_set = self.pos_data[pos_train_size : pos_train_size + pos_validate_size]
        pos_test_set = self.pos_data[pos_train_size + pos_validate_size : pos_total]
    
        self.neg_train_set = self.neg_data[0 : neg_train_size]
        neg_validation_set = self.neg_data[neg_train_size : neg_train_size + neg_validate_size]
        neg_test_set = self.neg_data[neg_train_size + neg_validate_size : neg_total]

        dec = len(neg_validation_set) - len(pos_validation_set)
        for _ in range(dec):
            pos_validation_set.append(random.choice(self.pos_data))

        dec = len(neg_test_set) - len(pos_test_set)
        for _ in range(dec):
            pos_test_set.append(random.choice(self.pos_data))

        self.validation_set = []
        self.validation_set.extend(pos_validation_set)
        self.validation_set.extend(neg_validation_set)

        self.test_set = []
        self.test_set.extend(pos_test_set)
        self.test_set.extend(neg_test_set)

    '''
        Ergodic files in dataset dir
    '''
    def lookup_dataset_dir(self, callback):
        for _, dirs, _ in os.walk(self.data_dir):
            for dir in dirs:
                for _, _, files in os.walk(os.path.join(self.data_dir, dir)):
                    for file in files:
                        callback(self, dir, file)

    '''
        Get iamge data
    '''
    def get_image_data(self, tp):
        image1, image2 = scm.imread(tp[0]), scm.imread(tp[1])

        newimg1 = np.array(scm.imresize(image1, (STD_HEIGHT, STD_WIDTH)))
        newimg2 = np.array(scm.imresize(image2, (STD_HEIGHT, STD_WIDTH)))
        
        # img_comb = np.hstack((newimg1, newimg2))[:, :, np.newaxis]

        img_comb = np.dstack((newimg1, newimg2))

        return img_comb / 255.0

    '''
        Get a batch of dataset
    '''
    def next_batch(self, batch_size):
        random_neg = batch_size // 2
        random_pos = batch_size - random_neg

        org_pos_data = []
        org_neg_data = []
        for _ in range(random_pos):
            org_pos_data.append(random.choice(self.pos_train_set))

        for _ in range(random_neg):
            org_neg_data.append(random.choice(self.neg_train_set))

        pos_data = list(map(self.get_image_data, org_pos_data))
        pos_labels = list(map(lambda e: e[2], org_pos_data))
        neg_data = list(map(self.get_image_data, org_neg_data))
        neg_labels = list(map(lambda e: e[2], org_neg_data))

        pos_data.extend(neg_data)
        pos_labels.extend(neg_labels)

        return np.array(pos_data), np.array(pos_labels)

    '''
        Get validation dataset
    '''
    def get_validation_set(self):
        data = np.array(list(map(self.get_image_data, self.validation_set)))
        labels = np.array(list(map(lambda e: e[2], self.validation_set)))
        
        return data, labels
        
    '''
        Get test dataset
    '''
    def get_test_set(self):
        data = np.array(list(map(self.get_image_data, self.test_set)))
        labels = np.array(list(map(lambda e: e[2], self.test_set)))
        return data, labels

# obj = DataSet('./Pic', 8)
# obj.check_image_channels()
# obj.load_dataset()
# obj.generate_dataset()

# data, labels = obj.next_batch(8)
# while done != True:
#     print(data[0][0].dtype)
#     data, labels, done = obj.next_batch()