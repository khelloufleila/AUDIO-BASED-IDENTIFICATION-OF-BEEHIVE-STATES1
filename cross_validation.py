#from cross_validation import  get_list_samples_name_, cross_validation_4folds, constrainedsplit
# Khellouf leila 
# utility functions
# python -m pip install SoundFile

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
#tf.random.set_seed(seed_value)
# for later versions: 
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
#___________________________________________________________________________________________________
import glob
import os
from info import i, printb, printr, printp, print
import glob
import os
import librosa
import pdb
import csv
import json
import re
import numpy as np
import random
import librosa.display
import IPython.display as ipd
from sklearn import preprocessing
from collections import Counter
from matplotlib import pyplot as plt
from info import i, printb, printr, printp, print
import muda
import jams
from sklearn import svm
import librosa
import keras
import scipy.io as sio
import io
from scipy.io import savemat
from scipy.sparse import csr_matrix
from os.path import dirname, join as pjoin
#_______________________________________________________________
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
#from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from keras.models import Sequential, Input, Model 
from keras.layers import Dense, Dropout, Flatten, Activation 
from keras.layers import Conv2D , MaxPooling2D
from keras.layers.normalization import BatchNormalization 
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model
from keras.initializers import normal
from tensorflow.keras import layers
from tensorflow.keras import initializers
from sklearn.metrics import accuracy_score 
import pandas as pd 
import csv
import numpy as np
import scipy as sc
#________________________________________________________________
import timbre_descriptor as td
import scipy as sc
import my_tools as mt
from collections import namedtuple
import scipy
import scipy.signal
from scipy.io import wavfile

import matplotlib
import swipep as swp             # used for sing le-F0 estimation
import warnings                 # used for warning removal
import time               # used performance benchmark


#__________________________________________4 folds _Cross Validation ______________________________________________________________
# return the 4 folds : fold1= CF001+ CF003, fold2= CJ001+GH001, .....
def get_list_samples_name_( mtx , path_audioSegments, extension='.mat' ):
    states=['active','missing queen','swarm' ]
    labels1=[]
    labels2=[]
    labels3=[]
    labels4=[]
    Y1=[]
    Y2=[]
    Y3=[]
    Y4=[]
    sample_ids1=[]
    sample_ids2=[]
    sample_ids3=[]
    sample_ids4=[]
    ruche1=[]
    ruche2=[]
    ruche3=[]
    ruche4=[]
    
    for x in glob.glob(path_audioSegments+'*'+extension): 
        size=len(path_audioSegments)
        sample=x[size:]
        #print(sample)
        if sample[0:5]=="CF001" or sample[0:5]=="CF003":
            
            sample_ids1.append(sample)
            l= read_HiveState_fromSampleName( sample, states)
            labels1.append(l)
            m=scipy.io.loadmat(x)
            ruche1.append(m[mtx])
        elif sample[0:5]=="CJ001" or sample[0:5]=="GH001" :
            
            sample_ids2.append(sample)
            l= read_HiveState_fromSampleName( sample, states)
            labels2.append(l)
            m=scipy.io.loadmat(x)
            ruche2.append(m[mtx])
        elif sample[0:5]=="Hive1": 
            
            sample_ids3.append(sample)
            l= read_HiveState_fromSampleName( sample, states)
            labels3.append(l)
            m=scipy.io.loadmat(x)
            ruche3.append(m[mtx])
        else : 
            sample_ids4.append(sample)
            l= read_HiveState_fromSampleName( sample, states)
            labels4.append(l)
            m=scipy.io.loadmat(x)
            ruche4.append(m[mtx])
    
    Y1= labels2binary('active', labels1)
    Y2= labels2binary('active', labels2)
    Y3= labels2binary('active', labels3)
    Y4= labels2binary('active', labels4)
    return ruche1,Y1,labels1, sample_ids1, ruche2,Y2,labels2, sample_ids2, ruche3,Y3,labels3, sample_ids3, ruche4,Y4,labels4, sample_ids4
#ruche1,Y1,labels1, sample_ids1, ruche2,Y2,labels2, sample_ids2, ruche3,Y3,labels3, sample_ids3, ruche4,Y4,labels4, sample_ids4=get_list_samples_name_MFCCS('b', path_save_audio_MFCCs)

# Define the 4 Fold cross validation 
def cross_validation_4folds(fold, ruche1,Y1, ruche2,Y2, ruche3,Y3, ruche4,Y4, sample_ids1 , sample_ids2 , sample_ids3 , sample_ids4):
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    sample_ids_train=[]
    sample_ids_test=[]
    if fold==1:
        print("fold==1")
        x_train=ruche2+ ruche3+ ruche4
        y_train=Y2+ Y3+ Y4
        x_test=ruche1
        y_test=Y1
        sample_ids_train=sample_ids2+sample_ids3+sample_ids4
        sample_ids_test=sample_ids1
    elif fold==2:
        print("fold==2")
        x_train=ruche1 + ruche3+ ruche4
        y_train=Y1+ Y3+ Y4
        x_test=ruche2
        y_test=Y2
        sample_ids_train=sample_ids1+sample_ids3+sample_ids4
        sample_ids_test=sample_ids2
    elif fold==3:
        print("fold==3")
        x_train=ruche1 + ruche2+ ruche4
        y_train=Y1+ Y2+ Y4
        x_test=ruche3
        y_test=Y3
        sample_ids_train=sample_ids1+sample_ids2+sample_ids4
        sample_ids_test=sample_ids3
    else:
        print("fold==4")
        x_train=ruche1 + ruche2+ ruche3
        y_train=Y1+ Y2+ Y3
        x_test=ruche4
        y_test=Y4
        sample_ids_train=sample_ids1+sample_ids2+sample_ids3
        sample_ids_test=sample_ids4
        
    return  x_train, x_test, y_train, y_test, sample_ids_train, sample_ids_test    

        
#x_train, x_test, y_train, y_test, sample_ids_train, sample_ids_test= cross_validation_4folds(3, ruche1,Y1, ruche2,Y2, ruche3,Y3, ruche4,Y4, sample_ids1 , sample_ids2 , sample_ids3 , sample_ids4)  
#len(x_train), len(x_test), len(y_train), len(y_test), len(sample_ids_train)

# train_test_split the data we take 0.7% training and 0.3% for the test 

def constrainedsplit(y_train, x_test, y_test, ratio=0.7):
	
	nb_class = len(np.unique(y_test));   ## nb de classes
	
	N_max_test = round(len(y_train)/ratio * (1-ratio))
	
	N_test  = np.zeros(nb_class);
	N_train = np.zeros(nb_class);
	for i in range(nb_class):
		N_test[i]  = round(len(y_test[y_test==i]))
		N_train[i] = round(len(y_train[y_train==i]))
	
	I = np.argsort(N_test);
	for i in range(nb_class):
		j = I[i]
		
		N_max_tmp = round(N_train[j] / ratio * (1-ratio));
		
		n_tmp = int(min( N_test[j], N_max_tmp))
		x_tmp = x_test[y_test==j,:]
		y_tmp = y_test[y_test==j]
		if i == 0:
			x_test_out = x_tmp[0:n_tmp, :]
			y_test_out = y_tmp[0:n_tmp]
		else:
			x_test_out = np.concatenate([x_test_out, x_tmp[0:n_tmp, :]])
			y_test_out = np.concatenate([y_test_out, y_tmp[0:n_tmp]])
	
	return x_test_out,y_test_out


def split_samples_ramdom(test_size, train_size, path_audioSegments_folder, splitPath_save, extension ='.wav'):

    
    splittedSamples = {'test': [], 'train': [], 'val':[]}
    
    list_samples_id = get_list_samples_names(path_audioSegments_folder, extension)
    n_segments = len(list_samples_id)
    samplesTEST=random.sample(list_samples_id, round(n_segments*test_size))
    samples_rest=np.setdiff1d(list_samples_id , samplesTEST)
    samplesVAL=random.sample(samples_rest.tolist(), round(samples_rest.size*train_size))
    samplesTRAIN=np.setdiff1d(samples_rest , samplesVAL).tolist()

    
    print('samples for testing: '+ str(list(samplesTEST)))
    print('samples for training: '+ str(list(samplesTRAIN)))
    print('samples for validation: '+ str(samplesVAL))
    
    splittedSamples['val'] = samplesVAL
    splittedSamples['test'] = samplesTEST
    splittedSamples['train'] = samplesTRAIN
    
    with open(splitPath_save+'.json', 'w') as outfile:
        json.dump(splittedSamples, outfile)
    
    return splittedSamples

    
def get_samples_id_perSet(pathSplitFile):  # reads split_id file

   
    split_dict=json.load(open (pathSplitFile, 'r'))
    
    sample_ids_test = split_dict['test'] 
    sample_ids_train = split_dict['train'] 
    sample_ids_val = split_dict['val']
    return sample_ids_test, sample_ids_train, sample_ids_val  


#________________________________split by hive_____________________________________________________________________________
def write_sample_ids_perHive(sample_ids , savepath):
    
  #identify different hives:
            #in the NU-Hive dataset the hives are identified in the string Hive1 or Hive3 in the beginning.
            #in OSBH every file referring to the same person will be considered as if the same Hive: identified by #nameInitials -
            # other files that do not follow this can be grouped in the same hive ()
            #get from unique filenames all unique identifiers of hives: either read the string until the first_  : example 'Hive3_
            #or get the string starting in '#' until the first ' - '. example: '#CF003 - '
    #uniqueFilenames=['Hive3_20_07_2017_QueenBee_H3_audio___15_40_00.wav', 
    #                 'Hive1_20_07_2017_QueenBee_H3_audio___15_40_00.wav', 
    #                 'Hive3_20_07_22017_QueenBee_H3_audio___15_40_00.wav', 
    #                 'Sound Inside a Swarming Bee Hive  -25 to -15 minutes-sE02T8B2LfA.wav', 
    #                 'Sound Inside a Swarming Bee Hive  +25 to -15 minutes-sE02T8B2LfA.wav', 
    #                 '#CF003 - Active - Day - (222).csv', '#CF003 - Active - Day - (212).csv']
    
    
    uniqueHivesNames={}
    pat1=re.compile("(\w+\d)\s-\s")
    pat2=re.compile("^(Hive\d)_")
    pat3=re.compile("^(Hive\d)\s")
    for sample in sample_ids:

        match_pat1=pat1.match(sample)
        match_pat2=pat2.match(sample)
        match_pat3=pat3.match(sample)
        
        if match_pat1:
            if match_pat1.group(1) in uniqueHivesNames.keys():
                uniqueHivesNames[match_pat1.group(1)].append(sample)
            else: 
                uniqueHivesNames[match_pat1.group(1)]=[sample]
        
        
        elif match_pat3:
            if match_pat3.group(1) in uniqueHivesNames.keys():
                uniqueHivesNames[match_pat3.group(1)].append(sample)
            else: 
                uniqueHivesNames[match_pat3.group(1)]=[sample]
              
        elif match_pat2:
            if match_pat2.group(1) in uniqueHivesNames.keys():
                uniqueHivesNames[match_pat2.group(1)].append(sample)
            else: 
                uniqueHivesNames[match_pat2.group(1)]=[sample]
        else: 
            #odd case, like files names 'Sound Inside a Swarming Bee Hive  -25 to -15 minutes-sE02T8B2LfA.wav'
            #will be all gathred as the same hive, although we need to be careful if other names appear!
            if 'Sound Inside a Swarming Bee Hive' in uniqueHivesNames.keys():
                uniqueHivesNames['Sound Inside a Swarming Bee Hive'].append(sample)
            else: 
                uniqueHivesNames['Sound Inside a Swarming Bee Hive']=[sample]  
                
    
    
    with open(savepath+'sampleID_perHive.json', 'w') as outfile:
        json.dump(uniqueHivesNames, outfile)
    
    return uniqueHivesNames




def split_samples_byHive(test_size, train_size, hives_data_dictionary, splitPath_save):
    
    ## creates 3 different sets intended for hive-independent classification. meaning that samples are separated accordingly to the hive.
    ## input: test_size, ex: 0.1  : 10% hives for test
    ## train_size, ex: 0.7: 70% hives for training, 30% for valisdation. (after having selected test samples!!)  
    ## splitPath_save = path and name where to save the splitted samples id dictionary
    
    ## output:
    ## returns and dumps a dictionary: {test : [sample_id1, sample_id2, ..], train : [], 'val': [sample_id2, sample_id2]}
    
    splittedSamples={'test': [], 'train': [], 'val':[]}
    
    n_hives = len(hives_data_dictionary.keys())
    
    hives_list=list(hives_data_dictionary.keys())
        
    hives_rest1=random.sample(hives_list, round(n_hives*(1-test_size)))
    
    if len(hives_rest1) == len(hives_list):
        rand_hive = random.sample(range(len(hives_rest1)),1)
        hives_rest=hives_rest1[:]
        del hives_rest[rand_hive[0]]
    else:
        
        hives_rest = hives_rest1[:]
  
    hiveTEST=np.setdiff1d(hives_list , hives_rest)
    hiveVAL=random.sample(hives_rest, round(len(hives_rest)*train_size))
    hiveTRAIN=np.setdiff1d(hives_rest , hiveVAL)
    
    
    print('hives for testing: '+ str(list(hiveTEST)))
    print('hives for training: '+ str(list(hiveTRAIN)))
    print('hives for validation: '+ str(hiveVAL))
    
    
    for ht in list(hiveTEST):
        splittedSamples['test'].extend(hives_data_dictionary[ht])

    for h1 in list(hiveTRAIN):
        splittedSamples['train'].extend(hives_data_dictionary[h1])
    
    for h2 in hiveVAL:
        splittedSamples['val'].extend(hives_data_dictionary[h2])

    with open(splitPath_save+'.json', 'w') as outfile:
        json.dump(splittedSamples, outfile)
    
    return splittedSamples
