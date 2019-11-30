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

import Binary_Clf_v1 as Clf_v1



        #### MAIN EXECUTABLE #####

if __name__ == '__main__':

            #### INITIALIZE ####

    intdir = 'C:/Users/Landon/Documents/GitHub/FFT-Neural-Net/Binary_Classifier_v1'
    target = np.loadtxt('is_string.txt')        # load target dataset
    target = np.append(target,target)           # duplicate for L & R
    print("Loading Data...")

    L_Frame = pd.read_csv('L_FFT_Frame.txt',sep='\t',index_col=0)
    L_Frame = L_Frame.transpose()
    
    R_Frame = pd.read_csv('R_FFT_Frame.txt',sep='\t',index_col=0)
    R_Frame = R_Frame.transpose()

    Frame = pd.concat([L_Frame,R_Frame])    # concatenate dataframes
    print(Frame.info())                     # info about frames
    print("Data Loaded\n")

            #### SGD CLASSIFIER ####

    CLF,xy_data = Clf_v1.binary_classifier(Frame,target,1000)   # classifier & dictionary
    X_train = xy_data['X_train']        # x training data
    Y_train = xy_data['Y_train']        # y training data
    print("Built SGD Classifier")

    conf_mat,conf_dict = \
        Clf_v1.confusion(CLF,X_train,Y_train)        # build confusion matrix
    print(conf_mat)
