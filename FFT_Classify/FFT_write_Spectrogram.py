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
import FFT_Classify

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

            #### Specify Dir Paths & Collect Files ####
    intdir = os.getcwd()
    readdir = 'C:/Users/Landon/Documents/wav_audio/Violins' 
    wavs = FFT_Classify.read_directory(readdir) 
    print("Number of files to read in this path:",len(wavs))

    strings = ['Violins','Violas','Violoncellos']
    spects = np.array([])

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

            #### Create Spectrogram ####
        f,t,Sxx = audio.spectrogram('L',npts=N,ovlp=int(0.75*N))
        setattr(audio,'Sxx_flatten',Sxx.flatten())



