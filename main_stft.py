# Khellouf leila 
# utility functions
# python -m pip install SoundFile

#___________________________________________________________________________________________________
import glob
import os
#from info import i, printb, printr, printp
import glob
import os
#import librosa
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
from scipy.sparse import csc_matrix
#_______________________________________________________________
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import plot_precision_recall_curve
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


from libda import SNR, sigmerge, data_augmentation
from decomposition import read_beeNotBee_annotations_saves_labels,  load_audioFiles_saves_segments, uniform_block_size , write_Statelabels_from_beeNotBeelabels, read_HiveState_fromSampleName , get_list_samples_names
from classification import SVM_Classification_BeehiveSTATE,  deep_model, Dense_Net , train_and_evaluate_model
from cross_validation import  get_list_samples_name_, cross_validation_4folds, constrainedsplit
from feature_extraction import raw_feature_fromSample, get_features_from_samples, labels2binary, get_GT_labels_fromFiles
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

nbits = 16;
MAX_VAL = pow(2,(nbits-1)) * 1.0;
class_names= ['missing queen', 'active' ]

n_folds=4
epochs=50
batch_size=145

#----------------------------------- parameters to change-----------------------------------#
block_size=1 # blocks of 1 second
thresholds=[0, 0.5]  # minimum length for nobee intervals: 0 or 5 seconds (creates one label file per threshold value)
path= 'C:\\Users\PC\python\Stage\'
path_audioFiles= path+ "\To Bee or not to Bee_the annotated dataset"+os.sep  # path to audio files
annotations_path= path + "To Bee or not to Bee_the annotated dataset"+os.sep # path to .lab files
path_save_audio_labels= path+ 'dataset_BeeNoBee_2_second'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.

path_workingFolder= path+ 'dataset_BeeNoBee_2_second'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
labels2read= 'state_labels'
feature = 'MFCCs20'

path_working_stft= path+ 'dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\stft_matrix.mat'+os.sep
path_save_audio_stft= path+ 'dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\stft_matrix.mat'+os.sep 


nbits = 16;
MAX_VAL = pow(2,(nbits-1)) * 1.0;
target_names=['missing_queen', 'active']

#-----------------------------------STFT+CNN -----------------------------------#

ruche1,Y1,labels1, sample_ids1, ruche2,Y2,labels2, sample_ids2, ruche3,Y3,labels3, sample_ids3, ruche4,Y4,labels4, sample_ids4=get_list_samples_name_('b', path_save_audio_stft)
# save the model history in a list after fitting so we can plot later 
model_history=[]
val_accuracy=[]
for i in range(4):
    fold= i+1
    print("Training on Fold :", fold)
    x_train, x_test, y_train, y_test,sample_ids_train, sample_ids_test=cross_validation_4folds(fold, ruche1,Y1, ruche2,Y2, ruche3,Y3, ruche4,Y4 , sample_ids1 , sample_ids2 , sample_ids3 , sample_ids4) 
    print(len(x_train), len(x_test), len(y_train), len(y_test))
    # Read a sparse matrix 
    x_train2=[]
    x_test2=[]
    for l in range(len(x_train)):
        x_train2.append(x_train[l].todense())
    for l in range(len(x_test)):
        x_test2.append(x_test[l].todense())
        
    X_train= np.array(x_train2)
    x_Test= np.array(x_test2)
    y_train=np.array(y_train)
    y_test= np.array(y_test)
    X_test, y_test = constrainedsplit(y_train, x_Test, y_test, 0.7)
    print(X_train.shape,  X_test.shape, y_train.shape, y_test.shape)
    X_train=X_train[:,0:512, :]
    X_test=X_test[:, 0:512, :]
    print(X_train.shape,  X_test.shape, y_train.shape, y_test.shape)
    
    print("Reshape the data")
    
    x, y, z= X_train.shape
    X_train= X_train.reshape(-1, 512, 43, 1)
    Y_train=y_train.reshape(-1, 1)

    x, y, z= X_test.shape
    X_test= X_test.reshape(-1,  512, 43, 1)
    Y_test=y_test.reshape(-1, 1)
    
  
    print("labelencoder")
    # Encode the classification labels
    le = LabelEncoder()
    y_train = to_categorical(le.fit_transform(y_train)) 
    y_test = to_categorical(le.fit_transform(y_test))
    
    
    size=( 512,43, 1)
    model_filename = "stft_cnn_model_cpu_multifilter_fold{}.hdf5".format(fold)
    model=None
    model=deep_model(size)
    results, val_acc, report, confusion_matrix = train_and_evaluate_model(model, X_train, y_train, X_test, y_test,  Y_test ,epochs, batch_size, model_filename )
    model_history.append(results)
    val_accuracy.append(val_acc)
    df = pd.DataFrame(report).transpose()
    name="classification report for stft + CNN.csv "+str(fold)
    filename="confusion_matrix for stft + CNN "+str(fold)
    df.to_csv(name)
    save_confusion_matrix(confusion_matrix,filename )

    #print("============="*12, end="\n\n\n")
    
