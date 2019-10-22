"""
Landon Buell
FFT & Neural Network
Raw Audio files main (v1)
8 Sept 2019
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os
import FFT_Classify

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

            #### Specify Dir Paths & Collect Files ####
    readdir = 'C:/Users/Landon/Documents/wav_audio/Violoncellos' 
    wavs = FFT_Classify.read_directory(readdir) 
    print("Number of files to read in this path:",len(wavs))

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
        audio.normalize(['L','R'])              # normailize L&R waveform to 1
        audio.divisible_by_N(N)                 # make length of arrays divisible by N
        audio.crop_silence(['L'],N=N,bnd=0.2)   # elim dead noise
        audio.slicebyidx(['R','time'],\
            np.arange(0,len(audio.L)))          # slice each time array
        audio.divisible_by_N(N)                 # make length of arrays divisible by N

            #### Create Freqency Spectrum ####
        fspace,power = audio.Fast_Fourier_Transform(['L'])      # Compute FFTs
        audio.normalize(['L_FFT',])                             # normalize
        pts = np.where((fspace>=0)&(fspace<=5000))              # 0 to 5000 Hz
        audio.slicebyidx(['L_FFT','freq_space'],pts)            # slice attrbs

        os.chdir(wavs[I].fftpath)
        FFT_Classify.Plot_Freq(audio,['L_FFT'],save=True)

            #### Create Spectrogram ####
        hann = audio.hanning_window(N=N,M=len(audio.L)/N)   # create hannign wondow taper
        setattr(audio,'L',hann*audio.L)
        #f,t,Sxx = audio.spectrogram('L')            # comepute spectrogram    
        f,t,Sxx = audio.scipy_spectrogram('L',N=N)

        os.chdir(wavs[I].spectpath)
        FFT_Classify.Plot_Spectrogram(audio,'L_Sxx',save=True)