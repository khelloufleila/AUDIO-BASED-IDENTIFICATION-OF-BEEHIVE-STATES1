# Khellouf leila 
# utility functions
# python -m pip install SoundFile

#___________________________________________________________________________________________________

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
#import librosa.display
import IPython.display as ipd
from sklearn import preprocessing
from collections import Counter
from matplotlib import pyplot as plt
#from info import i, printb, printr, printp, print
#import muda
#import jams
from sklearn import svm
#import librosa
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
from plotting import save_confusion_matrix, plot_confusion_matrix, plot_accuracy_val_accuracy
from libda import SNR, sigmerge, data_augmentation
from decomposition import read_beeNotBee_annotations_saves_labels,  load_audioFiles_saves_segments, uniform_block_size , write_Statelabels_from_beeNotBeelabels, read_HiveState_fromSampleName , get_list_samples_names
from classification import SVM_Classification_BeehiveSTATE,  deep_model, Dense_Net , train_and_evaluate_model
from cross_validation import read_HiveState_fromSampleName, get_list_samples_name_, cross_validation_4folds, constrainedsplit
from feature_extraction import labels2binary, raw_feature_fromSample, get_features_from_samples, labels2binary, get_GT_labels_fromFiles
#________________________________________________________________
#from utils import get_list_samples_names, get_features_from_samples, write_Statelabels_from_beeNotBeelabels,raw_feature_fromSample, labels2binary , get_GT_labels_fromFiles, get_items2replicate, BalanceData_online, get_list_samples_name_MFCC, SVM_Classification_BeehiveSTATE , fit_and_evaluate ,deep_model, plot_confusion_matrix, get_list_samples_name_TTBOX, Dense_Net, plot_accuracy_val_accuracy, read_HiveState_fromSampleName ,train_and_evaluate_model , cross_validation_4folds, get_list_samples_name_ , constrainedsplit, save_confusion_matrix#________________________________________________________________

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


#----------------------------------- parameters to change-----------------------------------#
block_size=1 # blocks of 1 second
thresholds=[0, 0.5]  # minimum length for nobee intervals: 0 or 5 seconds (creates one label file per threshold value)
path= 'C:\\Users\PC\python\Stage'
path1= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'
path_audioFiles= path+ "\To Bee or not to Bee_the annotated dataset"+os.sep  # path to audio files
annotations_path= path+ "\To Bee or not to Bee_the annotated dataset"+os.sep # path to .lab files
path_save_audio_labels= path+ '\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
#----------------------------------- parameters to change-----------------------------------#
path_workingFolder= path+ '\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
labels2read= 'state_labels'
feature = 'MFCCs20'
path_working_MFCCs20= path1 +'\\MFCCs20_matrix.mat'+os.sep
path_working_TTBox  = path1 +'\\TTBox_matrix.mat'+os.sep
path_working_stft   = path1 +'\\stft_matrix.mat'+os.sep
path_working_cqt    = path1 +'\\cqt_matrix.mat'+os.sep

path_save_audio_MFCCs= path1 +'\\MFCCs20_matrix.mat'+os.sep 
path_save_audio_ttbox= path1 +'\\ttb_mat'+os.sep 
path_save_audio_stft = path1 +'\\stft_matrix.mat'+os.sep 
path_save_audio_cqt  = path1 +'\\cqt_matrix.mat'+os.sep 


nbits = 16;
MAX_VAL = pow(2,(nbits-1)) * 1.0;
target_names=['missing_queen', 'active']



ruche1,Y1,labels1, sample_ids1, ruche2,Y2,labels2, sample_ids2, ruche3,Y3,labels3, sample_ids3, ruche4,Y4,labels4, sample_ids4=get_list_samples_name_('b',path_save_audio_MFCCs)
for i in range(4):
    fold= i+1
    print("Training on Fold :", i+1)

    x_train, x_test, y_train, y_test,sample_ids_train, sample_ids_test=cross_validation_4folds(i+1, ruche1,Y1, ruche2,Y2, ruche3,Y3, ruche4,Y4 , sample_ids1 , sample_ids2 , sample_ids3 , sample_ids4) 
    print(len(x_train), len(x_test), len(y_train), len(y_test))
    
    #y_train, x_train, sample_ids1_train= BalanceData_online(y_train, x_train, sample_ids_train)
    #y_test, x_test, sample_ids1_test= BalanceData_online(y_test, x_test, sample_ids_test)
    # Convert features and corresponding classification labels into numpy arrays
    X_train = np.array(x_train)
    y_train = np.array(y_train)
    X_test2 = np.array(x_test)
    y_test2 = np.array(y_test)
    print(y_train.shape, X_train.shape, y_test2.shape, X_test2.shape)
    
    X_test, y_test = constrainedsplit(y_train, X_test2, y_test2, 0.7)
    x, y, z= X_train.shape
    x1, y1, z1= X_test.shape
    print("reshape the data: ")
    X_train= X_train.reshape(x, y*z)
    X_test= X_test.reshape(x1, y1*z1)
    print(y_train.shape, X_train.shape, y_test.shape, X_test.shape)
    CLF, Test_GroundT, Train_GroundT, Test_Preds, Train_Preds, Test_Preds_Proba, Train_Preds_Proba = SVM_Classification_BeehiveSTATE(X_train, y_train , X_test, y_test, kerneloption='rbf')

    # Metrics
    print("Accuracy: ", metrics.accuracy_score( Test_GroundT, Test_Preds))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(Test_GroundT, Test_Preds))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(Test_GroundT, Test_Preds))
   
    # Compute confusion matrix
    confusion_mat = confusion_matrix(Test_GroundT, Test_Preds )
    filename="confusion_matrix for MFCC + SVM "+str(fold)
    save_confusion_matrix(confusion_mat,filename, target_names )
    
    report=classification_report(Test_GroundT, Test_Preds , target_names=target_names, output_dict=True)
    name="classification report for MFCC + SVM.csv "+str(fold)
    df = pd.DataFrame(report).transpose()
    df.to_csv(name)
    
    
    
    print("============="*12, end="\n\n\n")
 
