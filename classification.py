
# from classification import SVM_Classification_BeehiveSTATE,  deep_model, Dense_Net , train_and_evaluate_model
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

from cross_validation import  get_list_samples_name_, cross_validation_4folds, constrainedsplit

nbits = 16;
MAX_VAL = pow(2,(nbits-1)) * 1.0;
target_names=['missing_queen', 'active']




#_____________________________________________________  SVM functions_________________________________________________________



# SVM CLASSIFICATION:
    

def SVM_Classification_BeehiveSTATE(X_flat_train, y_train, X_flat_test, y_test, kerneloption='rbf'):

    print('\n')
    printb('Starting classification with SVM:')
    Test_Preds=[]
    Train_Preds=[]
    Test_Preds_Proba=[]
    Train_Preds_Proba=[]
    Test_GroundT=[]
    Train_GroundT=[]
   
    print('\n')
    printb('classification Beehive State into : Active or Missing Queen')
        
    #train :
    CLF = svm.SVC(kernel=kerneloption, probability=True)
    CLF.fit(X_flat_train, y_train)
    y_pred_train = CLF.predict(X_flat_train)
    y_pred_proba_train = CLF.predict_proba(X_flat_train)
    
    Train_GroundT = y_train
    Train_Preds = y_pred_train
    Train_Preds_Proba = y_pred_proba_train[:,1]
    
    # test:
    y_pred_test = CLF.predict(X_flat_test)
    y_pred_proba_test = CLF.predict_proba(X_flat_test)
    Test_GroundT= y_test
    Test_Preds = y_pred_test
    Test_Preds_Proba = y_pred_proba_test[:,1]

    return CLF, Test_GroundT, Train_GroundT, Test_Preds, Train_Preds, Test_Preds_Proba, Train_Preds_Proba 

#_____________________________________________________CNN Classification __________________________________________

def deep_model(size):
    # Neural Network Architecture 
    model=Sequential()
    # size= ( 20,44, 1)
   # for i in range(1, num_cnn_layers+1):
         
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=size , padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.add(BatchNormalization())
   # model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    # we add the dropout to skip the overfitting 
    # model.add(Dropout(0.25))
    # add the batch_normalization



    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
   # model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))

    # we add the dropout to skip the overfitting 
    # model.add(Dropout(0.25))

    model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    #model.add(Dropout(0.25))

    model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))

    # we add the dropout to skip the overfitting 
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(32 , activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model









#______________________________________TTBOX+DenseNet_____________________________________________________________________
def Dense_Net(size): 
    # size=(164,1 )
    # Create the model
    model=Sequential()
    # the first Dense layer
    model.add(Flatten(input_shape=size))
    model.add(Dense(328 ,  activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # the second dense layer 
    model.add(Dense(328 , activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # the third dense layer 
    model.add(Dense(328,  activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # the output layer 
    model.add(Dense(2, activation="softmax" ,  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model


# Fit and evaluate the model 

def train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test, y_test ,  nb_epoch, batch_size ,model_filename):
    # y_test is labels, Y_test is categorical labels
    
    print('Train...')
    target_names=['missing_queen', 'active']
    stopping = EarlyStopping(monitor='val_accuracy', patience=50)
    checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1, save_best_only=True ,save_weights_only=False, monitor='val_accuracy' , mode='max')
    results= model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[stopping, checkpointer], verbose=2 , validation_data=(X_test, Y_test))
    # prediction
    best_model = load_model(model_filename)
    probabilities = best_model.predict(X_test, batch_size=batch_size)
    predictions = probabilities.argmax(axis=-1)
    best_acc = accuracy_score(y_test, predictions)
    print('Accuracy score (best model): {}'.format(best_acc))
    #print("classification report: ", classification_report(y_test, predictions))
    report = classification_report(y_test, predictions , target_names=target_names, output_dict=True)
    cnf_matrix = confusion_matrix(y_test, predictions )
    return results, best_acc, report, cnf_matrix 












