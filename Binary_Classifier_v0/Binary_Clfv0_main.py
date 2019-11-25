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
import Binary_Clfv0 as Clfv0


            #### MAIN EXECUTABLE FUNCTION ####

if __name__ == '__main__':

    """
    Program Outline:
    ----------------
    - Collect All Frequency Spectra data as an N x M array     
            Each row is the spectrum for a particular audio file
            Put name of file into array of names
            T/F array indicating if strings or not
    - Feed data into SGD Classifier
    """

    FFT_dir = 'C:/Users/Landon/Documents/wav_FFTs'


    os.chdir(FFT_dir)
    files = Clfv0.read_directory(FFT_dir)
    print("Number of Files:",len(files))
    print("Collecting Data...")
    L_FFT_Frame = pd.DataFrame()

    """ Classifying By Frequency Spectra """

    for file in files:                              # It. through list of objs 
        os.chdir(file.dirpath)                      # move to directory path
        frame = Clfv0.Read_CSV(file.file,cols=[2])  # read the CSV file
        L_FFT_Frame[str(file.file)] = frame         # add to large data frame

            #### We Now Have all files sotred into a DataFrame #### 

        FFT_Training,FFT_Testing = Clfv0.model.train_test_split(L_FFT_Frame,
                                                          test_size=0.1,random_state=42)
            
        


    
     