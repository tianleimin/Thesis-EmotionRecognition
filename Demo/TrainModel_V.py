# train Valence prediction model with uttAVEC+IEMOCAP
# FL fusion: audio (opensmile-eGeMAPS) + lexical (CSA)
# feel free to play with different recognition models and learning techniques

from __future__ import print_function
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import h5py
import os
os.environ['KERAS_BACKEND']='theano'
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Merge, Input, BatchNormalization, Flatten, Reshape
from keras.optimizers import RMSprop,Adamax
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from keras import backend as K
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# turn off the warnings, be careful when use this
import warnings
warnings.filterwarnings("ignore")

# save outputs to a log file in case there is a broken pipe
import sys
idlestdout = sys.stdout
logger = open("output/output_V.txt", "w")
sys.stdout = logger

# data files
A_emo = 'data/uttAVEC_biAV.csv'
A_aud = 'data/uttAVEC_GeMAPS.csv'
A_lex = 'data/uttAVEC_CSA.csv'
I_emo = 'data/IEMOCAP_biAV.csv'
I_aud = 'data/IEMOCAP_GeMAPS.csv'
I_lex = 'data/IEMOCAP_CSA.csv'

# meta parameters
nb_aud_feat = 88 # dimensionality of feature set 1
nb_lex_feat = 63 # dimensionality of feature set 2
nb_feat = nb_aud_feat + nb_lex_feat # total number of features, also the number of neurons in the input layer of LSTM

time_step = 5  # the length of history (number of previous data instances) to include
batch_size = 32
nb_epoch = 1000 # number of total epochs to train the model
# if the validation loss isn't decreasing for a number of epochs, stop training to prevent over-fitting
early_stopping = EarlyStopping(monitor='val_loss', patience=100)

opt_func = Adamax(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # optimization function

# function to reshape the panda.DataFrame format data to Keras style: (batch_size, time_step, nb_features)
def reshape_data(data, n_prev = time_step):
    docX = []
    for i in range(len(data)):
        if i < (len(data)-n_prev):
            docX.append(data.iloc[i:i+n_prev].as_matrix())
        else: # the frames in the last window use the same context
            docX.append(data.iloc[(len(data)-n_prev):len(data)].as_matrix())
    alsX = np.array(docX)
    return alsX

# read in data
A_aud_feat = pd.read_csv(A_aud, header=None)
A_lex_feat = pd.read_csv(A_lex, header=None)

I_aud_feat = pd.read_csv(I_aud, header=None)
I_lex_feat = pd.read_csv(I_lex, header=None)

# normalize features
A_aud_feat = (A_aud_feat - A_aud_feat.min())/(A_aud_feat.max() - A_aud_feat.min())
A_lex_feat = (A_lex_feat - A_lex_feat.min())/(A_lex_feat.max() - A_lex_feat.min())
I_aud_feat = (I_aud_feat - I_aud_feat.min())/(I_aud_feat.max() - I_aud_feat.min())
I_lex_feat = (I_lex_feat - I_lex_feat.min())/(I_lex_feat.max() - I_lex_feat.min())

# combine the two databases
trn_aud_feat = pd.concat([A_aud_feat,I_aud_feat])
trn_lex_feat = pd.concat([A_lex_feat,I_lex_feat])

# FL fusion
all_feat_trn = pd.concat([trn_aud_feat, trn_lex_feat], axis=1)

X = reshape_data(all_feat_trn)

A_emo_raw = pd.read_csv(A_emo, header=None, usecols=[1])
I_emo_raw = pd.read_csv(I_emo, header=None, usecols=[1])
trn_emo_raw = pd.concat([A_emo_raw,I_emo_raw])
trn_emo = trn_emo_raw.values
y = np.asarray(trn_emo)

# save a subset of IEMOCAP for testing
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, shuffle=False)

print("Data preprocessing finished! Begin compiling and training model.")

# Building FL fusion model
model = Sequential()
model.add(LSTM(64, input_shape=(time_step, nb_feat), dropout_W=0.2, dropout_U=0.2, return_sequences=False))
model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True))
model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True))
model.add(Dense(32, activation='relu', W_regularizer=l2(0.0), trainable=True))
model.add(Dense(1, activation='sigmoid')) # output layer
model.compile(loss='binary_crossentropy', optimizer=opt_func, metrics=['binary_accuracy']) # define the optimizer for training

# training
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, y_valid), callbacks=[early_stopping], verbose=2)

# evaluation
model.evaluate(X_valid, y_valid, batch_size=batch_size)

# output predictions
np.set_printoptions(threshold=np.nan)
tst_pred = model.predict(X_valid)
print('\nPrinting predictions...')
tst_pred_file = "output/pred_V.csv"
tst_df = pd.DataFrame(tst_pred)
tst_df.to_csv(tst_pred_file, index=False, header=False)

# print confusion matrix
y_test_non_category = [ np.argmax(t) for t in y_valid ]
y_predict_non_category = [ np.argmax(t) for t in tst_pred ]
print('Confusion Matrix')
print(confusion_matrix(y_test_non_category, y_predict_non_category))
tst_f1 = f1_score(y_test_non_category, y_predict_non_category, average='weighted')
print('Test F1-score:', tst_f1)

# Save model and test with new data

# save model as JSON
model_json = model.to_json()
with open("pre-trained/V_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("pre-trained/V_model.h5")
print("Saved model to disk")

# Flush outputs to log file
logger.flush()
logger.close()
