"""
Landon Buell
Kevin Short
Binary Classifier - v0
5 Novemeber 2019
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
        self.file = file                        # filename

            #### FUNCTIONS DEFINTIONS ####

def read_directory(dir):
    """
    Read all files in given directory path
    --------------------------------
    dir (str) : desired directory path
    --------------------------------
    returns list of all '.txt' files
    """
    file_objs = []                          # list to '.txt' hold files objs
    for roots,dirs,files in os.walk(dir):   # all objects in the path
        for file in files:                  # files in list of files
            if file.endswith('.txt'):       # if '.txt' file
                wav = file_obj(roots,file)  # make instance
                file_objs.append(wav)       # add to list of file objs
    return file_objs                        # return lists  of files

def Read_CSV(filename,cols=[0,1,2,3]):
    """ Read Data From CSV File into pandas dataframe """
    data = np.loadtxt(filename,dtype=float,delimiter='\t',
                      skiprows=1,usecols=cols)
    return data