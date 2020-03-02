# Attention-BLSTM emotion classification (Arousal)
# Use DN features, train and test on AVEC (5-fold CV for each parameter set)

# import the required modules
from __future__ import print_function
import pandas as pd
import numpy as np
import csv

import os; os.environ['KERAS_BACKEND'] = 'theano'

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.optimizers import Adamax

from sklearn.metrics import confusion_matrix,f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,LabelEncoder
from sklearn.utils import class_weight

# Attention implemented by https://github.com/CyberZHG/keras-self-attention
from keras_self_attention import SeqSelfAttention

# define variables
# parameters to be investigated in grid seearch
time_steps = [2,4,8]  # input history to include, candidates: [2,4,8]
lstm_sizes = [[16,8,4],[32,16,8],[64,32,16]] # number of neurons in the BLSTM layers, candidates: [[16,8,4],[32,16,8],[64,32,16]]
attention_widths = [8,16,32] # width of the local context for the attention layer, candidates: [8,16,32]
# other parameters
batch_size = 32 # for estimating error gradient
nb_features = 5 # number of features
nb_class = 3 # number of classes
nb_epoch = 1000 # number of total epochs to train the model

# optimization function
opt_func = Adamax(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
# to prevent over-fitting
early_stopping = EarlyStopping(monitor='loss', patience=10)

# data files
file_log = '/Outputs/AVEC_DN_A_log.txt'
file_emo = '/Data/utt_AVEC_emo.csv'
file_feat = '/Data/utt_AVEC_DIS-NV.csv'

# turn off the warnings, be careful when use this
import warnings
warnings.filterwarnings("ignore")

# reshape panda.DataFrame to Keras style: (batch_size, time_step, nb_features)
def reshape_data(data, n_prev):
    docX = []
    for i in range(len(data)):
        if i < (len(data)-n_prev):
            docX.append(data.iloc[i:i+n_prev].values)
        else: # the frames in the last window use the same context
            docX.append(data.iloc[(len(data)-n_prev):len(data)].values)
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
data_feat = pd.read_csv(file_feat, header=None)
# normalize features
data_feat = (data_feat - data_feat.min())/(data_feat.max() - data_feat.min())

data_emo_raw = pd.read_csv(file_emo, header=None, usecols=[0])
data_emo_raw = data_emo_raw.values
# one-hot encoding of the classes
data_emo = []
for label in data_emo_raw:
    if label == 0:
        converted_label = [0,1,0] # medium
    elif label == 1:
        converted_label = [0,0,1] # high
    else:
        converted_label = [1,0,0] # low
    data_emo.append(converted_label)
data_emo = np.asarray(data_emo)
data_emo_df = pd.DataFrame(data_emo, index=None)

# Grid search for best parameters
para_list = []
#tst_pred_list = []
f1_list = []
count = 1
for time_step in time_steps:
    X = reshape_data(data_feat, time_step) # pad feature data
    y = reshape_data(data_emo_df, time_step) # pad label data
    for lstm_size in lstm_sizes:
        for attention_width in attention_widths:
            para_list.append([time_step, lstm_size, attention_width]) # save parameter set
            print('\n================================ No. %s of 27 ========================================' % count)
            print('\nParameters: time_step = %s, [h1, h2, h3] = %s, attention_width = %s\n' % (time_step, lstm_size, attention_width))
            # build model with given parameters
            model = attBLSTM(lstm_size, attention_width, nb_class, opt_func)
            # compile the model
            model.compile(loss='categorical_crossentropy', optimizer=opt_func, metrics=['categorical_accuracy'])
	
            # 5-fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            cv_num = 1
            cv_f1 = []
            for train, test in skf.split(X, np.zeros(shape=(X.shape[0], 1))):
                print("\nTraining on fold " + str(cv_num) + "/5...")
                cv_num = cv_num + 1
                # Generate batches from indices
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]
                # training the model
                model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
                # evaluation
                model.evaluate(X_test, y_test, batch_size=batch_size)
                # print confusion matrix
                y_test_non_category = [ np.argmax(t[0]) for t in y_test ]
                tst_pred = model.predict(X_test)
                y_predict_non_category = [ np.argmax(t[0]) for t in tst_pred ]
                print('Confusion Matrix on test set')
                print(confusion_matrix(y_test_non_category, y_predict_non_category))
                tst_f1 = f1_score(y_test_non_category, y_predict_non_category, average='weighted')
                cv_f1.append(tst_f1) # save f1 score
                print('Weighted F1-score on test set:', tst_f1)
				
            # Compute average F1-score for all cv folds
            f1_mean = sum(cv_f1) / float(len(cv_f1))
            print('\n=====Average F1-score: %s=====' % f1_mean)
            f1_list.append(f1_mean)

            # print grid search log
            with open(file_log, 'a') as logfile:
                logfile.write('\n================================ No. %s of 27 ========================================\n' % count)
                logfile.write('CV-F1 = %s; Parameters: time_step = %s, [h1, h2, h3] = %s, attention_width = %s\n' % (f1_mean, time_step, lstm_size, attention_width))          
            count = count + 1

# save the best parameter set and its predictions
best = f1_list.index(max(f1_list)) # find the highest F1 score
result = f1_list[best]
para = para_list[best]
#prediction = tst_pred_list[best]
with open(file_log, 'a') as logfile:
    logfile.write('\n================================ Best Performance ========================================\n')
    logfile.write('F1 = %s; Parameters: time_step = %s, [h1,h2,h3] = %s, attention_width = %s\n' % (result, para[0], para[1], para[2]))

print('\nDone!')
