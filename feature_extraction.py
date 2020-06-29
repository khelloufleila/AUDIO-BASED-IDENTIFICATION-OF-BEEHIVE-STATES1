# from feature_extraction import raw_feature_fromSample, get_features_from_samples, labels2binary, get_GT_labels_fromFiles
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

#______________________________________________________________________________________________________________
#----------------------------------- parameters to change-----------------------------------#
block_size=1 # blocks of 1 second
thresholds=[0, 0.5]  # minimum length for nobee intervals: 0 or 5 seconds (creates one label file per threshold value)
path_audioFiles="C:\\Users\PC\python\Stage\To Bee or not to Bee_the annotated dataset"+os.sep  # path to audio files
annotations_path="C:\\Users\PC\python\Stage\To Bee or not to Bee_the annotated dataset"+os.sep # path to .lab files
path_save_audio_labels= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
#----------------------------------- parameters to change-----------------------------------#
path_workingFolder='C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
labels2read= 'state_labels'
feature = 'MFCCs20'
path_working_MFCCs20= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\MFCCs20_matrix.mat'+os.sep
path_working_TTBox= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\TTBox_matrix.mat'+os.sep
path_working_stft= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\stft_matrix.mat'+os.sep
path_working_cqt= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\cqt_matrix.mat'+os.sep


path_save_audio_MFCCs= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\MFCCs20_matrix.mat'+os.sep 
path_save_audio_ttbox= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\ttb_mat'+os.sep 
path_save_audio_stft= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\stft_matrix.mat'+os.sep 
path_save_audio_cqt= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\cqt_matrix.mat'+os.sep 

nbits = 16;
MAX_VAL = pow(2,(nbits-1)) * 1.0;
target_names=['missing_queen', 'active']


#_____________________________________Calculate the features ____________________________________________________________________    


def raw_feature_fromSample( path_audio_sample, feature2extract ):
        
    s, Fs = librosa.core.load(path_audio_sample) # sr= 22050
    #m = re.match(r"\w+s(\d+)", feature2extract) #  m:  <re.Match object; span=(0, 7), match='MFCCs20'>
    #n_freqs=int(m.groups()[0]) ## n_freqs= 20
    #Melspec = librosa.feature.melspectrogram(audio_sample, n_mels = n_freqs) # computes mel spectrograms from audio sample, 
    
    if 'MFCCs' in feature2extract:
        n_freqs = int(feature2extract[5:len(feature2extract)]) ## n_freqs==20
        Melspec = librosa.feature.melspectrogram(s, sr=Fs)
        x = librosa.feature.mfcc(S=librosa.power_to_db(Melspec),sr=Fs, n_mfcc = n_freqs)
        #x = librosa.feature.mfcc(s ,Fs=Fs, n_mfcc = n_freqs)
        #print(x.shape)
    # Short-time Fourier transform    
    elif 'stft' in feature2extract:
          x= np.abs(librosa.stft(s , n_fft=2048, hop_length=514, win_length=2048, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
        
    # constante Q transform  Using a higher frequency resolution  
    elif 'cqt' in  feature2extract:
        x = np.abs(librosa.cqt(s, sr=Fs, fmin=librosa.note_to_hz('C2'), n_bins=60 * 2, bins_per_octave=12 * 2))
        
    # timber tool box     
    elif 'TTBOX' in feature2extract:
        r_Fs = librosa.resample(s, Fs, 44100)
        r_Fs=44100
        s= s/MAX_VAL;
        N = len(s);
       
        desc_TEE, desc_AS, desc_STFTmag, desc_STFTpow, desc_Har, desc_ERB, desc_GAM =td.compute_all_descriptor(s, r_Fs);
        desc  = [desc_TEE, desc_AS, desc_STFTmag, desc_STFTpow, desc_Har, desc_ERB, desc_GAM];

        ### Time serie integration
        param_val, field_name = td.temporalmodeling(desc);
        x=param_val
    
        
    else:
        x = Melspec

    return x 



def get_features_from_samples(path_audio_samples, sample_ids, raw_feature, normalization, high_level_features ): 
    #normalization = NO, z_norm, min_max
    ## function to extract features 
    #high_level_features = 0 or 1 
    #file_path = os.path.isfile(path_save_audio_labels+'matrix.mat'+'.csv') 
    n_samples_set = len(sample_ids) # 4
    feature_Maps = []
    if raw_feature== 'MFCCs20': 
         path_working= path_working_MFCC20
    elif  raw_feature=='TTBOX': 
         path_working= path_working_TTBox
    elif raw_feature=='stft':
         path_working= path_working_stft
    else: 
        path_working= path_working_cqt
    
    for sample in sample_ids:
        # raw feature extraction:
        print("sample: ",sample)
        x = raw_feature_fromSample( path_audio_samples+sample, raw_feature ) # x.shape: (4, 20, 2584)
       # print("x.shape: ",x.shape)
       # Sauvgarder les x dans un fichier .mat 
     
        b = csr_matrix(x)
        savemat(path_working+ sample[:-4] + '.mat', {'b': b})
         
                         
                    
        ##normalization here:si on veut les résultats pour Conv1D on utlise la normalisation 
        ##normalization here:
        if not normalization == 'NO':
             x_norm = featureMap_normalization_block_level(x, normalizationType = normalization) 
        else: x_norm = x
        
        if high_level_features:
            # high level feature extraction:
            if 'MFCCs' in raw_feature:
                X = compute_statistics_overMFCCs(x_norm, 'yes') # X.shape: (4 , 120)
            else: 
                X = compute_statistics_overSpectogram(x_norm)
                
            feature_map=X
        else:
            feature_map=x_norm
        
        
        feature_Maps.append(feature_map)
        
    return feature_Maps

def labels2binary(pos_label, list_labels):  # pos_label = missing queen / nobee
    list_binary_labels=[]
    for l in list_labels:
        #print("l=", l, "pos_label= ", pos_label)
        if l == pos_label:
            list_binary_labels.append(1)
        else:
            list_binary_labels.append(0)
    return list_binary_labels

def get_GT_labels_fromFiles(path_labels, sample_ids, labels2read) : #labels2read =  name of the label file    
    
    ##reads labels files and retreives labels for a sample set given by sample_ids
    # input:  path_labels: where label file is 
    #         sample_ids: list of sample names that we want the label
    #         labels2read: name of the labels file: state_labels, labels_BeeNotBee_th0 ...
    
    # output: list of string labels, in same order as sample_ids list
    
    labels = []
    fileAsdict={}
    
    with open(path_labels + labels2read+'.csv', 'r') as labfile:
        csvreader = csv.reader(labfile, delimiter=',')    
        for row in csvreader:
            if not row[0] == 'sample_name':
               # print("row[0]",row[0]) # CF001 - Missing Queen - Day -__segment0
               # print("row[-1]:", row[-1])  # bee or nobee
                fileAsdict[row[0]]=row[-1]   # row[-1] = '/missing queen/active' or 'bee/nobee'
            
    for sample in sample_ids:
        # print(sample)
        #print(fileAsdict[sample[0:-4]]) # bee nobee or unknown 
        labels.append(fileAsdict[sample[0:-4]])  #remove .wav extension
    
       
    return labels  
