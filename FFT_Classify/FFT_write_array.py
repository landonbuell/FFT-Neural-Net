"""
Landon Buell
FFT & Neural Network
Write Spectrogram
5 November 2019
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import FFT_Classify

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

            #### Specify Dir Paths & Collect Files ####
    intdir = os.getcwd()
    readdir = 'C:/Users/Landon/Documents/wav_audio' 
    #FFT_dir = 'C:/Users/Landon/Documents/wav_FFTs'
    #WF_dir = 'C:/Users/Landon/Documents/wav_Waveforms'
    #SPECT_dir = 'C:/Users/Landon/Documents/wav_spectrograms'

    wavs = FFT_Classify.read_directory(readdir) 
    print("Number of files to read in this path:",len(wavs))

    strings = ['Violins','Violas','Violoncellos']
    
    frequency_dataframe = pd.DataFrame()
    is_string = np.array([])

    N = (2**10)                  # N cols in reshaped array

    for I in range (len(wavs)):
        
            #### Move to Working Directory ####
        wavs[I].make_paths()                    # make needed paths
        os.chdir(wavs[I].dirpath)               # change to reading dir      
        name = str(wavs[I].file)                # create name for file
        print('File name:',name)
        print('\tFile size:',os.path.getsize(name),'bytes')
        
            #### Read Raw .Wav Data ####
        rate,data = FFT_Classify.read_wav(name) # read raw .wav file
        params = [2,'None']

            #### Create Class Instance ####
        audio = FFT_Classify.audio_sample(name,data,rate,params) 
        
            #### WORKING WITH FFT ###
        fspace,power = audio.Fast_Fourier_Transform(['L','R'],n=2**18)  # Compute FFTs
        audio.normalize(['L_FFT','R_FFT'])                      # normalize
        pts = np.where((fspace>=0)&(fspace<=4000))              # 0 to 5000 Hz
        audio.slicebyidx(['L_FFT','R_FFT','freq_space'],pts)    # slice attrbs
        


