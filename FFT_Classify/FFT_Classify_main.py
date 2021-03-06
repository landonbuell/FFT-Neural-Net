"""
Landon Buell
FFT & Neural Network
Raw Audio files main (v1)
8 Sept 2019
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import FFT_Classify

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

            #### Specify Dir Paths & Collect Files ####
    intdir = os.getcwd()
    readdir = 'C:/Users/Landon/Documents/wav_audio' 
    outdir = 'C:/Users/Landon/Documents/GitHub/FFT-Neural-Net/Binary_Classifier_v0'
    wavs = FFT_Classify.read_directory(readdir) 
    print("Number of files to read in this path:",len(wavs))

    N = (2**10)                         # N cols in reshaped array
    cntr = 0                            # files in set
    strings = ['Violin','Viola','Cello']
    is_string = np.array([])            # array to hold T/F
    
    for I in range (len(wavs)):
        
            #### Move to Working Directory ####
        wavs[I].make_paths()                # make needed paths
        os.chdir(wavs[I].dirpath)           # change to reading dir      
        name = str(wavs[I].file)            # create name for file
        print('File name:',name)
        print('\tFile size:',os.path.getsize(name),'bytes')
        print('\tTime:',time.process_time())
        
            #### Read Raw .Wav Data ####
        rate,data = FFT_Classify.read_wav(name)     # read raw .wav file
        params = [2,'None']                         # additional params

            #### Create Class Instance ####
        audio = FFT_Classify.audio_sample(name,data,rate,params) 
        audio.normalize(['L','R'])                      # normailize L&R waveform to 1
        pts = np.arange(int(2**10),int(2**16))          # section of audiofile
        try:                                            # attempt
            audio.slicebyidx(['L','R','time'],slice=pts)# slice

        except:                                         # if error
            continue                                    # skip iteration

            #### Working with Wavefoms ####
        os.chdir(wavs[I].wavepath)                              # change dirpath
        audio.to_csv(name=audio.name,attrs=['time','L','R'])    # write attrs to csv
        FFT_Classify.Plot_Time(audio,title=audio.name+'.L',
                               attrs=['L'],save=True)           # plot attrs (and save)
        FFT_Classify.Plot_Time(audio,title=audio.name+'.R',
                               attrs=['R'],save=True)           # plot attrs (and save)

            #### Create Freqency Spectrum ####
        os.chdir(wavs[I].fftpath)                                       # change dirpath
        fspace,power = audio.Fast_Fourier_Transform(['L','R'],n=2**18)  # Compute FFTs
        audio.normalize(['L_FFT','R_FFT'])                              # normalize
        pts = np.where((fspace>=0)&(fspace<=4000))                      # 0 to 5000 Hz
        audio.slicebyidx(['L_FFT','R_FFT','freq_space'],pts)            # slice attrbs 
        audio.to_csv(audio.name,
                     attrs=['freq_space','L_FFT','R_FFT'])              # write FFT data to csv
        FFT_Classify.Plot_Freq(audio,title=audio.name+'.L',
                               attrs=['L_FFT'],save=True) # plot & save to dir
        FFT_Classify.Plot_Freq(audio,title=audio.name+'.R',
                               attrs=['R_FFT'],save=True) # plot & save to dir

            #### Create Spectrogram ####
        os.chdir(wavs[I].spectpath)                                     # change dirpath
        f,t,Sxx = audio.spectrogram('L',npts=N,ovlp=int(0.75*N))        # create L spectrogram
        FFT_Classify.Plot_Spectrogram(audio,title=audio.name+'.Left',
                                      Sxx=audio.L_Sxx,t=audio.L_Sxx_t,
                                      f=audio.L_Sxx_f,save=True)        # plot & save to dir
        np.savetxt(audio.name+'left_spectrogram.txt',
                   X=np.array(Sxx,dtype=float))                         # write 2D arr to dir
        f,t,Sxx = audio.spectrogram('R',npts=N,ovlp=int(0.75*N))        # create R spectrogram
        FFT_Classify.Plot_Spectrogram(audio,title=audio.name+'.Right',
                                      Sxx=audio.R_Sxx,t=audio.R_Sxx_t,
                                      f=audio.R_Sxx_f,save=True)        # plot & save to dir
        np.savetxt(audio.name+'right_spectrogram.txt',
                   X=np.array(Sxx,dtype=float))                         # write 2D arr to dir

            #### Determine if string or not ####
        if audio.instrument in strings:
            is_string = np.append(is_string,True)
        else:
            is_string = np.append(is_string,False)

            #### Last Housekeeping ####
        cntr += 1

    os.chdir(outdir)
    np.savetxt('is_string.txt',X=np.array(is_string,dtype=int))         # save T/F data to CSV
    print("Total Process Time",time.process_time())