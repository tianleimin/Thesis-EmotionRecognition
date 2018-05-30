# Emotion recognition demo, predict binary Arousal and Valence
# FL fusion: audio (opensmile-eGeMAPS) + lexical (CSA)

# There are pre-trained models and test feature files included for you to try it out
# You can extract features from your own wav files and speech transcripts using FeatExtract.py
# You can replace the saved models. If you do remember to adapt feature extraction accordingly
# For training the emotion recognition models see TrainModel_A.py and TrainModel_V.py

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import os
os.environ['KERAS_BACKEND']='theano'
from keras.models import model_from_json

# turn off the warnings, be careful when use this
import warnings
warnings.filterwarnings("ignore")

time_step = 5  # the length of history (number of previous data instances) to include

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

# load json and create model
json_file_A = open('pre-trained/A_model.json', 'r')
A_model_json = json_file_A.read()
json_file_A.close()
A_model = model_from_json(A_model_json)
# load weights into new model
A_model.load_weights("pre-trained/A_model.h5")

json_file_V = open('pre-trained/V_model.json', 'r')
V_model_json = json_file_V.read()
json_file_V.close()
V_model = model_from_json(V_model_json)
# load weights into new model
V_model.load_weights("pre-trained/V_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
# read in test data
file_aud_tst = 'test/en_4065_test_eGeMAPS.csv'
file_lex_tst = 'test/en_4065_test_CSA.csv'
tst_aud_feat = pd.read_csv(file_aud_tst, header=None)
tst_lex_feat = pd.read_csv(file_lex_tst, header=None)

### normalize
##with open('pre-trained/norm.pkl') as f:
##    aud_min, aud_max, lex_min, lex_max = pickle.load(f)
##tst_aud_feat = (tst_aud_feat - aud_min)/(aud_max - aud_min)
##tst_lex_feat = (tst_lex_feat - lex_min)/(lex_max - lex_min)
# normalize features
tst_aud_feat = (tst_aud_feat - tst_aud_feat.min())/(tst_aud_feat.max() - tst_aud_feat.min())
tst_lex_feat = (tst_lex_feat - tst_lex_feat.min())/(tst_lex_feat.max() - tst_lex_feat.min())

# FL fusion
all_feat_tst = pd.concat([tst_aud_feat, tst_lex_feat], axis=1)
# expand the length by 5 times
all_feat_mult_tst = pd.concat([all_feat_tst, all_feat_tst, all_feat_tst, all_feat_tst, all_feat_tst], axis=0)
demo_test = reshape_data(all_feat_mult_tst)

# using the saved model to predict emotion of the test data
demo_pred_A = A_model.predict(demo_test)
demo_pred_V = V_model.predict(demo_test)

print("Predictions on the test data:")
# show emoji of binary Arousal and Valence
# make new figure with 2 subfigures
# each subfigure can have an image in it
fig = plt.figure()
image1 = plt.subplot(121)
image2 = plt.subplot(122)

# for a more serious set of emoji change PsheenEmotion to BoringEmoji
V1 = mpimg.imread('PusheenEmotion/Smile.png')
V0 = mpimg.imread('PusheenEmotion/Sad.png')
A1 = mpimg.imread('PusheenEmotion/Awake.png')
A0 = mpimg.imread('PusheenEmotion/Asleep.png')

# emotion of the whole recording is the average of all sentences
if sum(demo_pred_A)/len(demo_pred_A) < 0.5:
    Arousal = 'Arousal: Calm'
    _ = image1.imshow(A0)
else:
    Arousal = 'Arousal: Active'
    _ = image1.imshow(A1)

if sum(demo_pred_V)/len(demo_pred_V) < 0.5:
    Valence = 'Valence: Negative'
    _ = image2.imshow(V0)
else:
    Valence = 'Valence: Positive'
    _ = image2.imshow(V1)

# hide axis and show window with images
image1.axis("off")
image1.set_title(Arousal)
image2.axis("off")
image2.set_title(Valence)
plt.show()
