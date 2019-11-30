"""
Landon Buell
Kevin Short
Binary Classifier v1
30 November 2019
"""
   
        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import sklearn.metrics as metrics
import sklearn.model_selection as model
from sklearn.linear_model import SGDClassifier

        #### CLASS OBJECTS ####

class file_obj ():
    """ Create Raw wavefile object """
    def __init__(self,root,file):
        """ Initialize Class Object """
        self.dirpath = root                     # intial storage path
        self.filename = file                    # filename

        #### FUNCTIONS DEFINITIONS #####

def read_directory(dir):
    """ Read all files in given directory path """
    file_objs = []                          # list to '.txt' hold files objs
    for roots,dirs,files in os.walk(dir):   # all objects in the path
        for file in files:                  # files in list of files
            if file.endswith('.txt'):       # if '.txt' file
                wav = file_obj(roots,file)  # make instance
                file_objs.append(wav)       # add to list of file objs
    return file_objs                        # return lists  of files

def read_CSV (filename,hdr=0):
    """ Read CSV file w/ Pandas module """
    data = pd.read_csv(filename,sep='\t',header=hdr,
                       dtype=float)
    return data

def binary_classifier (xdata,ydata,max_iter,seed=0,size=0.1):
    """
    Create a Binary Classifier Object w/ sklearn
    ----------------
    xdata (array/DataFrame) : base dataset
    ydata (array/DataFrame) : target labels for dataset
    max_iter (int) : Maximum iterations for SGD operations
    seed (int) : Random state seed for shuffeling data (0 by default)
    size (float) : Relative size of testing data set (0.1 by default)
    ----------------
    Returns classifier object and dictionary of training/testing data
    """
    X_train,X_test = model.train_test_split(xdata,test_size=size,random_state=seed)
    Y_train,Y_test = model.train_test_split(ydata,test_size=size,random_state=seed)

    xy_dict =   {'X_train':X_train,'X_test':X_test,
                 'Y_train':Y_train,'Y_test':Y_test}     # train/test data into dictionary

    CLF = SGDClassifier(random_state=seed)    # create classifer object
    CLF.fit(X_train,Y_train)                # fit dataset
   
    return CLF,xy_dict                      # return classifier & xy data dictionary

def confusion (clf,xdata,ydata):
    """
    Build Confusion matric and dictionary for binary classifier
    ----------------
    clf (classifier obj) : Classifier object to build confusion matrix for
    xdata (array/DataFrame) : x-training dataset
    ydata (array/DataFrame) : y-training target dataset
    ----------------
    Returns Binary confusion matrix and dictionary of entries
    """
    ypred = model.cross_val_predict(clf,xdata,ydata)    # cross-val prediction
    conf_mat = metrics.confusion_matrix(ydata,ypred)    # build confusion matrix
    
    conf_dict = {'TN':conf_mat[0][0],'FP':conf_mat[0][1],
                 'FN':conf_mat[1][0],'TP':conf_mat[1][1]}   # confusion dictionary

    return conf_mat,conf_dict                           # return confusion matrix & dictionary

def general_metrics (clf,xdata,ydata,disp=True):
    """
    Build Confusion matric and dictionary for binary classifier
    ----------------
    clf (classifier obj) : Classifier object to build confusion matrix for
    xdata (array/DataFrame) : x-training dataset
    ydata (array/DataFrame) : y-training target dataset
    disp (bool) : Display outputs to command line (True by default)
    ----------------
    Returns Binary confusion matrix and dictionary of entries
    """
    ypred = model.cross_val_predict(clf,xdata,ydata)    # cross-val prediction
    
    precision = metrics.precision_score(ydata,ypred)    # compute precision score
    recall = metrics.recall_score(ydata,ypred)          # compute recall score
    f1 = metrics.f1_score(ydata,ypred)                  # compute f1 score

    if disp == True:                        # print output to line?
        print('Precision Score:',precision)
        print('Recall Score:',recall)
        print("F1 Score:",f1)

    return precision,recall,f1                  # return values
