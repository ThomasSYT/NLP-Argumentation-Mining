import numpy as np
np.random.seed(42)
import math
import random
import matplotlib.pyplot as plt
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras import metrics
from keras.utils import plot_model
from pandas import DataFrame

from random import choice

from sklearn.metrics import f1_score,accuracy_score,precision_score

from . import ArgMiningModel
from .data_reader import load_dataset, load_dataset_test


def get_index_dict(input_data):
    """
    Create index - word/label dict from list input
    @params : List of lists
    @returns : Index - Word/label dictionary
    """
    result = dict()
    vocab = set()
    i = 1
    # Flatten list and get indices
    for element in [word for sentence in input_data for word in sentence]:
        if element not in vocab:
            result[i]=element
            i+=1
            vocab.add(element)
    return result

class ArgMiningBiLSTM(ArgMiningModel):
    def __init__(self, params):
        super(ArgMiningBiLSTM, self).__init__(params)

    def train_and_predict(self):
        hidden_units = self.params["hidden_units"]
        dropout = self.params["dropout"]
        batch_size = self.params["batch_size"]
        model_path = self.params["model_path"]
        predict_file = self.params["prediction_path"]


        # Load data: [[token11, token12, ...],[token21,token22,...]]
        # and label: [[label11, label12, ...],[label21,label22,...]]
        X_train_data,y_train_data = load_dataset("train_en.txt")
        X_dev_data, y_dev_data = load_dataset("dev_en.txt")
        X_test_data = load_dataset_test("test_en_unlabeled.txt")

        #get embedding matrix and word_to_index
        embedding_file_en = 'embedding/glove.6B.300d.txt'
        embedding_matrix = np.zeros((400003,300))
        word_to_index = dict()
        i = 1
        with open (embedding_file_en, 'r') as f:
            lines = f.readlines()
            for line in lines:
                values = line.split()
                word = values[0]
                word_to_index[word] = i
                coefs = np.array(values[1:], dtype='float32')
                embedding_matrix[i] = coefs
                i += 1
            word_to_index['__PADDING__'.lower()] = 400001
            embedding_matrix[400001] = [0.0]*300
            word_to_index['__OOV__'.lower()] = 400002
            embedding_matrix[400002] =[random.uniform(-1,1) for j in range(300)]
        print('Loaded en_embedding %s word vectors.' % len(embedding_matrix))
        print('\n')
        index_to_word = dict((v,k) for k,v in word_to_index.items())

        # Get index -> label and label -> index dictionaries
        index_to_label = get_index_dict(y_train_data)
        label_to_index = dict((v,k) for k,v in index_to_label.items())

        # Get indexed data and labels - 400002 is the OOV index
        X_train_index = [[word_to_index.get(word.lower(), 400002) for word in sentence] for sentence in X_train_data]
        X_dev_index = [[word_to_index.get(word.lower(), 400002) for word in sentence] for sentence in X_dev_data]
        X_test_index = [[word_to_index.get(word.lower(), 400002) for word in sentence] for sentence in X_test_data]

        y_train_index = [[label_to_index[label] for label in sentence] for sentence in y_train_data]
        y_dev_index = [[label_to_index[label] for label in sentence] for sentence in y_dev_data]
    
    
        # For batch training:
        # Pad additional x=[400001]/y=[0] elements at the end for the last batch:
        X_train_padded = X_train_index + [[400001] for _ in range(math.ceil(len(X_train_index)/batch_size)*batch_size-len(X_train_index))]
        X_dev_padded = X_dev_index + [[400001] for _ in range(math.ceil(len(X_dev_index)/batch_size)*batch_size-len(X_dev_index))]
        X_test_padded = X_test_index + [[400001] for _ in range(math.ceil(len(X_test_index)/batch_size)*batch_size-len(X_test_index))]

        y_train_padded = y_train_index + [[0] for _ in range(math.ceil(len(y_train_index)/batch_size)*batch_size-len(y_train_index))]
        y_dev_padded = y_dev_index + [[0] for _ in range(math.ceil(len(y_dev_index)/batch_size)*batch_size-len(y_dev_index))]


        # Get maximum sentence length to pad instances
        max_sentence_length = max(map(lambda x: len(x),X_train_data+X_dev_data+X_test_data))

        # Get the number of classes:
        number_of_classes = len(index_to_label.items())


        # Pad for all inputs
        X_train = sequence.pad_sequences(X_train_padded, maxlen=max_sentence_length)
        X_dev = sequence.pad_sequences(X_dev_padded, maxlen=max_sentence_length, padding='post')
        X_test = sequence.pad_sequences(X_test_padded, maxlen=max_sentence_length, padding='post')

        # For categorical cross_entropy we need matrices representing the classes:
        # Note that we pad after doing the transformation into the matrix!
        y_train = sequence.pad_sequences(np.asarray([to_categorical(y_label,number_of_classes+1) for y_label in y_train_padded]), maxlen=max_sentence_length)
        y_dev = sequence.pad_sequences(np.asarray([to_categorical(y_label,number_of_classes+1) for y_label in y_dev_padded]), maxlen=max_sentence_length, padding='post')



        def build_model(embedding_matrix, max_sentence_length, number_of_classes):
            model = Sequential()
            model.add(Embedding(len(embedding_matrix),
                                len(embedding_matrix[0]),
                                input_length=max_sentence_length,
                                weights=[embedding_matrix],
                                mask_zero=True,
                                trainable=False,
                                batch_input_shape=(batch_size, max_sentence_length)))
            
            model.add(Dropout(dropout))
            model.add(Bidirectional(LSTM(hidden_units, input_shape = (batch_size,max_sentence_length), return_sequences=True)))
            model.add(Bidirectional(LSTM(hidden_units, input_shape = (batch_size,max_sentence_length), return_sequences=True)))
            model.add(Dropout(dropout))
            model.add(TimeDistributed(Dense(number_of_classes+1)))
            model.add(Activation('softmax'))
        
            model.compile('adam', 'categorical_crossentropy', metrics=[metrics.categorical_accuracy])
            return model
    
    
    
        class F1ScoreModelCheckpointer(Callback):
            def __init__(self, model):
                super(F1ScoreModelCheckpointer).__init__()
                self.model = model

            def on_train_begin(self, logs={}):
                self.best_f1 = 0.0
                self.f1s = 0.0
            
            def on_epoch_end(self, batch, logs={}):
                #1. Get predictions 
                predictions = self.model.predict(self.validation_data[0], batch_size=batch_size)
                # Compute F1 score for test set:
                test_pred = []
                test_truth = []
                #2. Flatten all outputs        
                for sent_pred, sent_truth in zip(predictions, self.validation_data[1]):
                    for lab, word_pred in zip(np.argmax(sent_truth,axis=1), np.argmax(sent_pred,axis=1)):
                        if lab != 0:    #remove padding
                            test_truth.append(index_to_label[lab])
                            test_pred.append(index_to_label.get(word_pred,max(test_truth)))
                #3. Compute f1 score 
                f1_test = f1_score(test_truth, test_pred, list(index_to_label.values()),average='macro')
                #4. Store best model
                self.f1s = f1_test
                if self.f1s > self.best_f1:
                    self.best_f1 = self.f1s
                    self.model.save_weights(model_path)

                print(" F1 score %f" % (self.f1s))
                return 
              
            
            
        def evaluate(model, raw_data, lab_data):
            best_f1 = 0.0
            f1s = 0.0
            #1. Get predictions 
            predictions = model.predict(raw_data, batch_size=batch_size)
            # Compute F1 score for test set:
            test_pred = []
            test_truth = []
            #2. Flatten all outputs        
            for sent_pred, sent_truth in zip(predictions, lab_data):
                for lab, word_pred in zip(np.argmax(sent_truth,axis=1), np.argmax(sent_pred,axis=1)):
                    if lab != 0:    #remove padding
                        test_truth.append(index_to_label[lab])
                        test_pred.append(index_to_label.get(word_pred,max(test_truth)))
            #3. Compute f1 score 
            f1_test = f1_score(test_truth, test_pred, list(index_to_label.values()),average='macro')
            #4. Store best model
            f1s = f1_test
            if f1s > best_f1:
                best_f1 = f1s
                model.save_weights(model_path)
            return f1s

        
        def train(model, raw_data, lab_data, np_epoch):
            checkpointer = F1ScoreModelCheckpointer(model)
            test_f1s = list()
            epoch = list()
            for i in range(np_epoch):
                print(i)
                model.fit(X_train,
                          y_train,
                          batch_size=batch_size,
                          epochs=1,#1
                          callbacks=[checkpointer],
                          validation_data=(raw_data, lab_data),
                          shuffle=False)
                model.reset_states()
                test_f1s.append(evaluate(model, raw_data, lab_data))
                epoch.append(i)
                model.reset_states()
                #checkpointer.loss_plot('epoch')
            history = DataFrame()
        
            history['epoch'] = epoch
            history['test'] = test_f1s
            return history

        def predict(model):
        
            model.load_weights(model_path)
            # Get class probabilities for the test set:
            predictions = model.predict(X_test, batch_size=batch_size)
            print('writting.......')
            # Compute F1 score for test set:
            test_pred = []
            #test_truth = []
            pre_data = []
            for sent_pred in predictions:
                for word_pred in sent_pred:
                    test_pred.append(index_to_label[word_pred.tolist().index(max(word_pred))])
                    if word_pred.tolist().index(max(word_pred)) == 0:
                        print("Warning, PADDING label got predicted!")
            #All sentences ending with ‘.’ are marked as ‘O’
            k = 0
            for i in range(len(X_test_data)):
                pre_data.append([])
                if X_test_data[i][-1] != '.':
                    for j in range(len(X_test_data[i])): 
                        pre_data[i].append([X_test_data[i][j],'O'])
                        k += 1
                else:
                    for j in range(len(X_test_data[i])):#All '.' are marked as 'O'.
                        if X_test_data[i][j] == '.':
                            pre_data[i].append([X_test_data[i][j],'O'])
                        else:
                            pre_data[i].append([X_test_data[i][j],test_pred[k]])
                        k += 1
            pre_label = []
            for i in pre_data:
                for j in i:
                    pre_label.append(j[1])      
            l = 0
            with open(predict_file, 'w') as f:
                for i in range(len(X_test_data)):
                    for j in range(len(X_test_data[i])):
                        f.write(str(X_test_data[i][j]))
                        f.write('_en')
                        f.write('\t')
                        f.write(pre_label[l])
                        f.write('\n')
                        l += 1
                    if i != len(X_test_data)-1:
                        f.write('\n')
            print('Write finished.\n')

        history = DataFrame()
        repeats = 1#5
        for i in range(repeats):
            np_epoch = self.params['epochs']
            model = build_model(embedding_matrix, max_sentence_length, number_of_classes)
            history = train(model, X_dev, y_dev, np_epoch)
            plt.plot(history['epoch'], history['test'], color='orange',label='f1 score')
        plt.savefig('epochs_diagnostic.png')
        plt.show()
        plot_model(model, to_file='model.png', show_shapes=True)
        predict(model)



