import numpy as np
np.random.seed(42)

import os


def load_dataset(filename, data_path="data"):
    i = 0
    y_train_data = []
    x_train_data = []
    #preprocessing data
    with open(os.path.join(data_path, filename),'r') as f:
            raw_data = f.readlines()
            #print(raw_data)
            x_train_data.append([])
            y_train_data.append([])
            for l in raw_data:
                if l != "\n":
                    word,label_ = l.split('_en\t')
                    label,_ = label_.split("\n")
                    x_train_data[i].append(word)
                    y_train_data[i].append(label)
                else:
                    x_train_data.append([])
                    y_train_data.append([])
                    i += 1
    return x_train_data,y_train_data


def load_dataset_test(filename, data_path="data"):
    i = 0
    x_test_data = []
    #preprocessing data
    with open(os.path.join(data_path, filename),'r') as f:
            raw_data = f.readlines()
            x_test_data.append([])
            for l in raw_data:
                if l != "\n":
                    word,_ = l.split('_en')
                    x_test_data[i].append(word)
                else:
                    x_test_data.append([])
                    i += 1
    return x_test_data
