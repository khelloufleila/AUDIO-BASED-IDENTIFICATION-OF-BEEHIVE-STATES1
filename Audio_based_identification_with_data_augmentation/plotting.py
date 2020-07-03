# from plotting import save_confusion_matrix , plot_confusion_matrix , plot_accuracy_val_accuracy


# Khellouf leila 
# utility functions
# python -m pip install SoundFile


import glob
import os
from info import i, printb, printr, printp, print
import glob
import librosa
import pdb
import csv
import json
import re
import numpy as np
import random
import librosa.display
import IPython.display as ipd
from collections import Counter
from matplotlib import pyplot as plt
from info import i, printb, printr, printp, print
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
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model
from sklearn.metrics import accuracy_score 
import pandas as pd 
import csv
#________________________________________________________________

class_names= ['missing queen', 'active' ]
target_names=['missing_queen', 'active']

#_______________________________________ For Plotting_________________________________________________________________________________
            
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_accuracy_val_accuracy(model_history):

# Plot to see how models are performing 

    plt.title('Accuracies vs Epochs ')
    plt.plot(model_history[0].history['accuracy'], label='Training Fold 1')
    plt.plot(model_history[1].history['accuracy'], label='Training Fold 2')
    plt.plot(model_history[2].history['accuracy'], label='Training Fold 3')
    plt.plot(model_history[3].history['accuracy'], label='Training Fold 4')
    plt.legend()
    plt.show()
    plt.title('Train Accuracy vs Val Accuracy')
    plt.plot(model_history[0].history['accuracy'], label='Train Accuracy Fold 1', color='black')
    plt.plot(model_history[0].history['val_accuracy'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
    plt.plot(model_history[1].history['accuracy'], label='Train Accuracy Fold 2', color='red', )
    plt.plot(model_history[1].history['val_accuracy'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
    plt.plot(model_history[2].history['accuracy'], label='Train Accuracy Fold 3', color='green', )
    plt.plot(model_history[2].history['val_accuracy'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
    plt.plot(model_history[3].history['accuracy'], label='Train Accuracy Fold 4', color='steelblue', )
    plt.plot(model_history[3].history['val_accuracy'], label='Val Accuracy Fold 4', color='steelblue', linestyle = "dashdot")



    plt.legend()
    plt.show()
    
def save_confusion_matrix(cnf_matrix, filename, class_names):
    fig = plt.figure()
    # Plot non-normalized confusion matrix

    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=filename)
    plt.savefig('confusion_matrix'+str(filename)+'.jpg')
    
    
    
    
    
    
    
    
#_____________________________________________________________________________________________________________




def get_list_samples_name_MFCC(path_audioSegments, extension='.mat'):
    states=['active','missing queen','swarm' ]
    X_ttbox=[]
    labels=[]
    Y=[]
    sample_ids=[]
    # Recupèrer tout les audios d'extention .wav"""""" glob.glob(path_audioSegments_folder+'*'+extension)""""""
    #list_mfcc=[os.path.basename(x) for x in glob.glob(path_audioSegments+'*'+extension)]
    
    for x in glob.glob(path_audioSegments+'*'+extension): 
        size=len(path_audioSegments)
        sample=x[size:]
        sample_ids.append(sample)
        l= read_HiveState_fromSampleName( sample, states)
        labels.append(l)
        m=scipy.io.loadmat(x)
        X_ttbox.append(m['b'])
    
    Y= labels2binary('active', labels)
        
    return X_ttbox,  labels , Y, sample_ids
def get_list_samples_name_TTBOX(path_audioSegments, extension='.mat'):
    states=['active','missing queen','swarm' ]
    X_ttbox=[]
    labels=[]
    Y=[]
    sample_ids=[]
    # Recupèrer tout les audios d'extention .wav"""""" glob.glob(path_audioSegments_folder+'*'+extension)""""""
    #list_mfcc=[os.path.basename(x) for x in glob.glob(path_audioSegments+'*'+extension)]
    
    for x in glob.glob(path_audioSegments+'*'+extension): 
        size=len(path_audioSegments)
        sample=x[size:]
        sample_ids.append(sample)
        l= read_HiveState_fromSampleName( sample, states)
        labels.append(l)
        m=scipy.io.loadmat(x)
        X_ttbox.append(m['ttb_vec'])
    
    Y= labels2binary('active', labels)
        
    return X_ttbox,  labels , Y, sample_ids



#_________________________________________________________________





#_____________________________________TTBOX+SVM________________________________________________________________________

def get_list_samples_name(path_audioSegments, extension='.mat'):
    X_ttbox=[]
    sample_ids=[]
    # Recupèrer tout les audios d'extention .mat"""""" glob.glob(path_audioSegments_folder+'*'+extension)""""""
    list_ttbox=[os.path.basename(x) for x in glob.glob(path_audioSegments+'*'+extension)]
    for x in glob.glob(path_audioSegments+'*'+extension):
        
        sample_ids.append(x[63:])
        m=scipy.io.loadmat(x)
        X_ttbox.append(m['ttb_vec'])
    return X_ttbox, sample_ids, list_ttbox





def fit_and_evaluate(train_x, val_x, train_y, val_y, EPOCHS=50, BATCH_SIZE=145 ):
    model=None
    model=deep_model(( 20,44, 1))
    results= model.fit(train_x, train_y, epochs=EPOCHS, batch_size= BATCH_SIZE, callbacks=[early_stopping, model_checkpoint], verbose=1, validation_split=0.1)
    print("Val Score :", model.evaluate(val_x, val_y))
    return results 

def BalanceData_online(y_set, x_set, sample_ids_set):
    
    ## balances already processed data (X and Y, just before classifier) by replicating samples of the least represented class.
    # input: y_set - binary labels of set, x_set - feature_maps of set, sample_ids_set - sample names in set, ( all have the same order!)
    # output: X, Y and sample_ids with replicated samples concatenated 
    
 
    printb( 'Balancing training data:' )
    print('will randomly replicate samples from least represented class')
    
    x2concatenate = x_set
    y2concatenate = y_set
    sample_ids2concatenate = sample_ids_set
    
    dict_items_replicate = get_items2replicate(y_set,sample_ids_set )
    #print("dict_items_replicate: ",dict_items_replicate)
    
    for i in range(len(sample_ids_set)):
        if sample_ids_set[i] in dict_items_replicate.keys() :
            
            sample_ids2concatenate =np.concatenate([sample_ids2concatenate, [sample_ids_set[i]]*dict_items_replicate[sample_ids_set[i]]])
            y2concatenate = np.concatenate([y2concatenate, [y_set[i]]*dict_items_replicate[sample_ids_set[i]]])
            x2concatenate = np.concatenate([x2concatenate, [x_set[i]]*dict_items_replicate[sample_ids_set[i]]])
            
    return y2concatenate, x2concatenate, sample_ids2concatenate


def get_items2replicate(list_Binary_labels, list_sample_ids):
    
    # get the samples to be replicated.
    # input: list of labels and sample_ids with same oreder!
    # ouptut: dictionary keys:name of samples to be replicated,  value: Number of times to replicate that sample.
    
    #assert( len(list_Binary_labels) - len(list_sample_ids) == 0), ('arguments should have the same number of elements)
    dict_items_replicate={}
    
    n_samples = len(list_Binary_labels)# 193
    #print("n_samples: ", list_Binary_labels)
    n_positive_labels = sum(list_Binary_labels)#158 = le nbr de 1
    #print("n_positive_labels: ",n_positive_labels)
    n_negative_labels = n_samples - n_positive_labels #35= le nbr de 0
    #print("n_negative_labels: ",n_negative_labels)
    
    pos_samples=[]
    neg_samples=[]
    
    for i in range(n_samples):
        if list_Binary_labels[i] == 1 :
            #print("list_sample_ids[i]= : ", list_sample_ids[i])
            pos_samples.append(list_sample_ids[i])
        else: 
            neg_samples.append(list_sample_ids[i])
            
    if n_positive_labels > n_negative_labels:
        #print(n_positive_labels, n_negative_labels)
        # Replicate negative samples as needed:
        dif=n_positive_labels-n_negative_labels
        items_replicate=random.choices(neg_samples, k=dif)
       # print("neg_samples= ",neg_samples, "items_replicate=",items_replicate)
 
    elif n_positive_labels < n_negative_labels:
        dif=n_negative_labels-n_positive_labels
        items_replicate=random.choices(pos_samples, k=dif)
              
    dict_items_replicate=Counter(items_replicate)
    #print(dict_items_replicate)
    return dict_items_replicate
