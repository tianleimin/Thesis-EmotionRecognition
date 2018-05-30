# extract CSA lexical features and eGeMAPS audio features from data

# inputs and outputs
# Sorry but I can't provide the original lex_input_f and aud_input_f due to data privacy concerns

# lex_input_f is a csv file that contains at least 3 columns: starttime, endtime, word
lex_input_f = 'test/en_4065.alignword.txt.full.csv' # transcript (word + word timings) of the test speech
aud_input_f = 'test/en_4065.wav' # wav recording of the test speech
CSA_dict_f = 'pre-trained/CRR_WordsAndRatings_0mean.csv' # affective lexicon dictionary
CSA_feat_f = 'test/en_4065_test_CSA.csv' # CSA feature output
ffmpeg_bin = 'ffmpeg' # where you install your ffmpeg
openSMILE_bin = 'opensmile-2.3.0/inst/bin/SMILExtract' # where you install your OpenSmile
config = 'opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf' # configuration file of OpenSmile
aud_output_f = 'test/en_4065_test_eGeMAPS.arff'
aud_feat_f = 'test/en_4065_test_eGeMAPS.csv' # eGeMAPS feature output

########################### Extracting CSA features ############################
# use nltk toolkit installed on Python 2.6
# Return sums of CSA scores over all words in an utterance as CSA features
# 63 features based on crowd-sourced annotation of Arousal, Power, Valence of English lemmas
# for unseen words, use 0

import csv
import numpy as np

# read in origianl transcripts with word and timings
import pandas as pd

df = pd.read_csv(lex_input_f, delim_whitespace=True)

startT = df['starttime']
endT = df['endtime']
words = df['word']

# pad words of every 5 seconds into an utterance
padded = [] # words of every 5 seconds as utterances
uttST = [] # start time of each utterance
uttET = [] # end time of each utterance

i = 0
while i < (len(words)-1):
    utterance = []
    j = 0
    uttST.append(float(startT[i]))
    for j in range(len(words)-i):
        if float(endT[i+j]) < (float(startT[i])+5.0):
            utterance.append(words[i+j])
        else:
            uttET.append(float(endT[i+j-1]))
            break
    padded.append(utterance)
    i += len(utterance)
uttET.append(float(endT.iloc[-1]))

# lemmatize all the words in transcript
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

lmtzr = WordNetLemmatizer()

# map treebank tags to wordnet tags
from nltk.corpus import wordnet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmas = []
# lemmatize each word in each utterance
for lines in padded:
    lines_changed = []
    for item in lines:
        # POS tagging
        item = item.lower()
        tagged = word_tokenize(item)
        tagged = pos_tag(tagged)
        # lemmatize first
        lemmatized = lmtzr.lemmatize(item,get_wordnet_pos(tagged[0][1]))        
        lines_changed.append(lemmatized)
    lemmas.append(lines_changed)

# remove stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')

cleaned = []
for roots in lemmas:
    cleaned.append([i.encode("utf-8") for i in roots if i not in stop])

# get CRR scores with the word as dictionary index and the list of scores as dictionary value
CRR = {}
reader = csv.reader(open(CSA_dict_f))
for row in reader:
    key = row[0]
    CRR[key] = row[1:]

# extract lex-CRR features
features = []
for instance in cleaned:
    # for each word in utterance, find its CRR score, use 0 for unseen word
    scores = []
    nums = []
    if instance == []:
        scores.append(CRR['meanscoresofall'])
    else:
        for word in instance:
            if word in CRR:
                scores.append(CRR[word])
            else:
                scores.append(CRR['meanscoresofall'])
    # transform scores to floats
    for item in scores:
        nums.append([float(i) for i in item])
    # get sum of each CRR score over all words in this utterance as CRR features
    features.append([sum(j) for j in zip(*nums)])

# normalize feature values with z-score normalization
ori_features = zip(*features)
norm_features = []

for original in ori_features:
    original_np = np.asarray(original)
    z_norm = (original_np - original_np.mean()) / original_np.std()
    normalized = ['%.6f'%(x) for x in z_norm]
    norm_features.append(normalized)

lex_CRR = zip(*norm_features)

# print to file
with open(CSA_feat_f, 'wb') as outfile:
    for output in lex_CRR:
        row = csv.writer(outfile, delimiter=',')
        row.writerow(output)

print('CSA features extracted!')

##################################################################################
############################# Extracting eGeMAPS features ########################
# extract eGeMAPS features from the wav file with openSMILE
import subprocess as sp
import sys
import time

# run ffmpeg to chop the whole recording by word timings
for i in range(len(uttST)):
    START_TIME = str(uttST[i])
    END_TIME = str(uttET[i])
    chopped = 'test/wav_chopped/en_4065_'+str(i)+'.wav'
    command1 = [ffmpeg_bin, '-i', aud_input_f, '-acodec', 'copy', '-ss', START_TIME, '-to', END_TIME, chopped]
    s1 = sp.Popen(command1,stderr=sp.PIPE)

# run openSMILE to extract eGeMAPS features from the chopped recordings
for i in range(len(uttST)):
    chopped_input_f = 'test/wav_chopped/en_4065_'+str(i)+'.wav'
    command2 = [openSMILE_bin,'-C',config,'-I',chopped_input_f,'-O',aud_output_f]
    s2 = sp.Popen(command2,stderr=sp.PIPE)
    outputline, error = s2.communicate()
    if not ('No output was written!' in error):
        time.sleep(0.01)
        pass

# tidy up and normalize the extracted eGeMAPS features
# remove header and useless columns
aud_feat = []
with open(aud_output_f,'r') as f:
    for _ in range(95):
        next(f)
    for line in f:
        line = line.split(',')
        aud_feat.append([float(i) for i in line[1:-1]])

# normalize feature values with z-score normalization
ori_aud_features = zip(*aud_feat)
norm_aud_features = []
for aud_original in ori_aud_features:
    aud_original_np = np.asarray(aud_original)
    z_norm_aud = (aud_original_np - aud_original_np.mean()) / aud_original_np.std()
    aud_normalized = ['%.6f'%(x) for x in z_norm_aud]
    norm_aud_features.append(aud_normalized)

aud_eGeMAPS= zip(*norm_aud_features)

# print to file
with open(aud_feat_f, 'wb') as outfile:
    for output in aud_eGeMAPS:
        row = csv.writer(outfile, delimiter=',')
        row.writerow(output)

print('eGeMAPS features extracted!')
