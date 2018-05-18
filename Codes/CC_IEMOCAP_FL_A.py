# LSTM emotion classification using GP+DIS-NV features, train on AVEC2012 and test on IEMOCAP

# import the required modules
from __future__ import print_function
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Merge, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adamax
from keras import backend as K
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.preprocessing import normalize,LabelEncoder
from sklearn.utils import class_weight

# define variables
time_step = 3  # the length of history (number of previous data instances) to include
batch_size = 10 # training in batches, won't influence the performance much
nb_epoch = 1000 # number of total epochs to train the model

H1 = 8 # number of neurons in the bottom hidden layer
H2 = 16 # number of neurons in the middle hidden layer
H3 = 8 # number of neurons in the top hidden layer
dropout_W1 = 0 # drop out weight (for preventing over-fitting) in H1
dropout_U1 = 0 # drop out weight (for preventing over-fitting) in H1
dropout_W2 = 0 # drop out weight (for preventing over-fitting) in H2
dropout_U2 = 0 # drop out weight (for preventing over-fitting) in H2
dropout_W3 = 0 # drop out weight (for preventing over-fitting) in H3
dropout_U3 = 0 # drop out weight (for preventing over-fitting) in H3

#opt_func = RMSprop(lr=0.00001) # training function
opt_func = Adamax(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # optimization function

# if the validation loss isn't decreasing for a number of epochs, stop training to prevent over-fitting
early_stopping = EarlyStopping(monitor='val_loss', patience=100)

nb_features = 8
# total number of features, also the number of neurons in the input layer

# data files
file_log = '/exports/csce/datastore/inf/groups/eddie_inf_hcrc_cstr_students/s1219694/TAC2018/Outputs/CC_IEMOCAP_GP+DN_A_log.txt'
file_pred = '/exports/csce/datastore/inf/groups/eddie_inf_hcrc_cstr_students/s1219694/TAC2018/Outputs/CC_IEMOCAP_GP+DN_A_pred.txt'
file_emo_trn = '/exports/csce/datastore/inf/groups/eddie_inf_hcrc_cstr_students/s1219694/TAC2018/utt_AVEC_emo.csv'
file_feat_trn = '/exports/csce/datastore/inf/groups/eddie_inf_hcrc_cstr_students/s1219694/TAC2018/utt_AVEC_GP+DN.csv'
file_emo_tst = '/exports/csce/datastore/inf/groups/eddie_inf_hcrc_cstr_students/s1219694/TAC2018/IEMOCAP_emo.csv'
file_feat_tst = '/exports/csce/datastore/inf/groups/eddie_inf_hcrc_cstr_students/s1219694/TAC2018/IEMOCAP_GP+DN.csv'

# turn off the warnings, be careful when use this
import warnings
warnings.filterwarnings("ignore")

# save outputs to a log file in case there is a broken pipe
import sys
idlestdout = sys.stdout
logger = open(file_log, "w")
sys.stdout = logger

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
trn_feat = pd.read_csv(file_feat_trn, header=None)
# normalize features
trn_feat = (trn_feat - trn_feat.min())/(trn_feat.max() - trn_feat.min())
X_train = reshape_data(trn_feat)
trn_emo_raw = pd.read_csv(file_emo_trn, header=None, usecols=[0])
trn_emo_raw = trn_emo_raw.values
# one-hot encoding of the classes
trn_emo = []
for label in trn_emo_raw:
    if label == 0:
        converted_label = [1,0,0] # low
    elif label == 1:
        converted_label = [0,1,0] # medium
    else:
        converted_label = [0,0,1] # high
    trn_emo.append(converted_label)
y_train = np.asarray(trn_emo)

# deal with the unbalanced class issue
# A dataframe that represents the categorical class of each one-hot encoded row
y_df = pd.DataFrame(y_train, index=None)
y_classes = y_df.idxmax(1, skipna=False)
# Instantiate the label encoder
le = LabelEncoder()
# Fit the label encoder to our label series
le.fit(list(y_classes))
# Create integer based labels Series
y_integers = le.transform(list(y_classes))
# Create dict of labels : integer representation
labels_and_integers = dict(zip(y_classes, y_integers))
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
sample_weights = class_weight.compute_sample_weight('balanced', y_integers)
class_weights_dict = dict(zip(le.transform(list(le.classes_)), class_weights))
# print(class_weights_dict)
# print(sample_weights)

tst_feat = pd.read_csv(file_feat_tst, header=None)
# normalize features
tst_feat = (tst_feat - tst_feat.min())/(tst_feat.max() - tst_feat.min())
X_test = reshape_data(tst_feat)
tst_emo_raw = pd.read_csv(file_emo_tst, header=None, usecols=[0])
tst_emo_raw = tst_emo_raw.values
# one-hot encoding of the classes
tst_emo = []
for label in tst_emo_raw:
    if label == 0:
        converted_label = [0,1,0] # medium
    elif label == 1:
        converted_label = [0,0,1] # high
    else:
        converted_label = [1,0,0] # low
    tst_emo.append(converted_label)
y_test = np.asarray(tst_emo)

# build model
model = Sequential()
model.add(LSTM(H1, input_shape=(time_step, nb_features), dropout_W=dropout_W1, dropout_U=dropout_U1, return_sequences=False)) # bottom hidden layer
#model.add(LSTM(H2, dropout_W=dropout_W2, dropout_U=dropout_U2, return_sequences=True)) # middle hidden layer
#model.add(LSTM(H3, dropout_W=dropout_W3, dropout_U=dropout_U3, return_sequences=False)) # top hidden layer
model.add(Dense(3, activation='softmax')) # output layer, predict emotion dimensions seperately
model.compile(loss='categorical_crossentropy', optimizer=opt_func, metrics=['categorical_accuracy']) # define the optimizer for training

# training, can also do class_weight=class_weights_dict
model.fit(X_train, y_train, sample_weight=sample_weights, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=2)

# evaluation
model.evaluate(X_test, y_test, batch_size=batch_size)

# output predictions
tst_pred = model.predict(X_test)
tst_df = pd.DataFrame(tst_pred)
tst_df.to_csv(file_pred, mode='a', index=False, header=False)

# print confusion matrix
y_test_non_category = [ np.argmax(t) for t in y_test ]
y_predict_non_category = [ np.argmax(t) for t in tst_pred ]
print('Confusion Matrix')
print(confusion_matrix(y_test_non_category, y_predict_non_category))
tst_f1 = f1_score(y_test_non_category, y_predict_non_category, average='weighted')
print('Test F1-score:', tst_f1)

# Flush outputs to log file
logger.flush()
logger.close()
