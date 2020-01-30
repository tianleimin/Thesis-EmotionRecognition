# -*- coding: utf-8 -*-

# Using Google Colab (Python 3 + TPU)
# Attention-BLSTM emotion classification (Arousal)
# Use GP+DIS-NV features, train on both databases and test on a subset of IEMOCAP

'''
To prevent Colab from time out, go to the google Colab console (ctrl+shift+i) and type:
function ClickConnect(){console.log("Working");document.querySelector("colab-toolbar-button#connect").click()}setInterval(ClickConnect,60000)

Don't exit the console until you get "Working" as the output in the console window. 
It would keep on clicking the page and prevent it from disconnecting.
Make sure you dont run anything for more than 12 hrs on Colab!
'''

# import the required modules
from __future__ import print_function
import pandas as pd
import numpy as np
import csv

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.optimizers import Adamax

from sklearn.metrics import confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,LabelEncoder
from sklearn.utils import class_weight

# Attention implemented by https://github.com/CyberZHG/keras-self-attention
!pip install keras-self-attention 
from keras_self_attention import SeqSelfAttention

# define variables
# parameters to be investigated in grid seearch
time_steps = [2,4,8]  # input history to include, candidates: [2,4,8]
lstm_sizes = [[16,8,4],[32,16,8],[64,32,16]] # number of neurons in the BLSTM layers, candidates: [[16,8,4],[32,16,8],[64,32,16]]
attention_widths = [8,16,32] # width of the local context for the attention layer, candidates: [8,16,32]
# other parameters
batch_size = 32 # for estimating error gradient
nb_features = 8 # number of features
nb_class = 3 # number of classes
nb_epoch = 1000 # number of total epochs to train the model

# optimization function
opt_func = Adamax(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
# to prevent over-fitting
early_stopping = EarlyStopping(monitor='loss', patience=10)

# data files
# access data from Google Drive
from google.colab import drive
drive.mount('/content/drive')
file_log = '/content/drive/My Drive/ColabAtt_example_log.txt'
file_pred = '/content/drive/My Drive/ColabAtt_example_pred.txt'
file_emo_tst = '/content/drive/My Drive/IEMOCAP_emo_seg.csv' # retained a subset for testing after parameter optimisation
file_feat_tst = '/content/drive/My Drive/IEMOCAP_GP+DN_seg.csv' # retained a subset for testing after parameter optimisation
file_emo_trn = '/content/drive/My Drive/utt_AVEC_emo.csv'
file_feat_trn = '/content/drive/My Drive/utt_AVEC_GP+DN.csv'

# turn off the warnings, be careful when use this
import warnings
warnings.filterwarnings("ignore")

# reshape panda.DataFrame to Keras style: (batch_size, time_step, nb_features)
def reshape_data(data, n_prev):
    docX = []
    for i in range(len(data)):
        if i < (len(data)-n_prev):
            docX.append(data.iloc[i:i+n_prev].as_matrix())
        else: # the frames in the last window use the same context
            docX.append(data.iloc[(len(data)-n_prev):len(data)].as_matrix())
    alsX = np.array(docX)
    return alsX

# define the BLSTM model with attention
def attBLSTM(lstm_size, attention_width, nb_class, opt_func):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=lstm_size[0], return_sequences=True))) # BLSTM layer 1
    model.add(Bidirectional(LSTM(units=lstm_size[1], return_sequences=True))) # BLSTM layer 2
    model.add(Bidirectional(LSTM(units=lstm_size[2], return_sequences=True))) # BLSTM layer 3
    model.add(SeqSelfAttention(attention_width=attention_width, attention_activation='sigmoid')) # attention layer
    model.add(Dense(units=nb_class, activation='softmax')) # output layer, predict emotion dimensions seperately
    return model

# read in data
trn_feat = pd.read_csv(file_feat_trn, header=None)
tst_feat = pd.read_csv(file_feat_tst, header=None)
# normalize features
trn_feat = (trn_feat - trn_feat.min())/(trn_feat.max() - trn_feat.min())
tst_feat = (tst_feat - tst_feat.min())/(tst_feat.max() - tst_feat.min())
# combine the two databases
all_feat = pd.concat([trn_feat,tst_feat])
trn_emo_raw = pd.read_csv(file_emo_trn, header=None, usecols=[0])
trn_emo_raw = trn_emo_raw.values
# one-hot encoding of the classes
trn_emo = []
for label in trn_emo_raw:
    if label == 0:
        converted_label = [0,1,0] # medium
    elif label == 1:
        converted_label = [0,0,1] # high
    else:
        converted_label = [1,0,0] # low
    trn_emo.append(converted_label)
y_train = np.asarray(trn_emo)
y_train_df = pd.DataFrame(y_train, index=None)
tst_emo_raw = pd.read_csv(file_emo_tst, header=None, usecols=[0])
tst_emo_raw = tst_emo_raw.values
# one-hot encoding of the classes
tst_emo = []
for label in tst_emo_raw:
    if label == 0:
        converted_label = [1,0,0] # low
    elif label == 1:
        converted_label = [0,1,0] # medium
    else:
        converted_label = [0,0,1] # high
    tst_emo.append(converted_label)
y_test = np.asarray(tst_emo)
y_test_df = pd.DataFrame(y_test, index=None)
# combine the two databases
all_emo_raw = pd.concat([y_train_df,y_test_df])

# Grid search for best parameters
para_list = []
tst_pred_list = []
f1_list = []
count = 1
for time_step in time_steps:
    X = reshape_data(all_feat, time_step) # pad feature data
    y = reshape_data(all_emo_raw, time_step) # pad label data
    # save a subset of IEMOCAP for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
    for lstm_size in lstm_sizes:
        for attention_width in attention_widths:
            para_list.append([time_step, lstm_size, attention_width]) # save parameter set
            print('\n================================ No. %s of 27 ========================================' % count)
            print('\nParameters: time_step = %s, [h1, h2, h3] = %s, attention_width = %s\n' % (time_step, lstm_size, attention_width))
            # build model with given parameters
            model = attBLSTM(lstm_size, attention_width, nb_class, opt_func)
            # compile the model
            model.compile(loss='categorical_crossentropy', optimizer=opt_func, metrics=['categorical_accuracy'])
            # training the model, use the last 5% as the validation set
            # remember to remove a subset of IEMOCAP/AVEC before reading in data as the seperate test set
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
                      validation_split=0.05, callbacks=[early_stopping], verbose=2)
            # evaluation
            model.evaluate(X_test, y_test, batch_size=batch_size)

            # save predictions
            tst_pred = model.predict(X_test)
            tst_pred_list.append(tst_pred) # save predictions

            # print confusion matrix
            y_test_non_category = [ np.argmax(t[0]) for t in y_test ]
            y_predict_non_category = [ np.argmax(t[0]) for t in tst_pred ]
            print('Confusion Matrix on test set')
            print(confusion_matrix(y_test_non_category, y_predict_non_category))
            tst_f1 = f1_score(y_test_non_category, y_predict_non_category, average='weighted')
            f1_list.append(tst_f1) # save f1 score
            print('Weighted F1-score on test set:', tst_f1)
            # print grid search log
            with open(file_log, 'a') as logfile:
                logfile.write('\n================================ No. %s of 27 ========================================\n' % count)
                logfile.write('F1 = %s; Parameters: time_step = %s, [h1, h2, h3] = %s, attention_width = %s\n' % (tst_f1, time_step, lstm_size, attention_width))
                logfile.write('Confusion Matrix on test set\n')
                np.savetxt(logfile, confusion_matrix(y_test_non_category, y_predict_non_category))           
            count = count + 1

# save the best parameter set and its predictions
best = f1_list.index(max(f1_list)) # find the highest F1 score
result = f1_list[best]
para = para_list[best]
prediction = tst_pred_list[best]
with open(file_pred, 'a') as predfile:
    predfile.write('F1 = %s; Parameters: time_step = %s, [h1,h2,h3] = %s, attention_width = %s\n' % (result, para[0], para[1], para[2]))
    for pred in prediction:
        indi_pred = []
        indi_pred = pred[0] # reform the seq prediction to individual samples
        row = ', '.join(map(str, indi_pred))
        predfile.write('%s\n' % row)

print('\nDone!')
