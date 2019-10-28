"""
Landon Buell
FFT & Neural Network
Read raw Audio files (v1)
8 Sept 2019
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.fftpack as fftpack
import scipy.io.wavfile as sciowav
import scipy.signal as signal

            #### CLASS OBJECTS ####

class audio_sample ():
    """
    Creates Audio sample object
    --------------------------------
    filename (str) : name of file that hold '.aif' data
    waveform (array) : n X m array of waveform data, n rows for n channels
    args (tuple) : truple of extra attributes for the object
    --------------------------------
    This audio object will be used as the default class object
    """

    def __init__(self,filename,waveform,rate,params):
        """ Initalize audio_sample class object instance """
            #### Given Attributes ####
        self.name = filename            # file name
        self.L = waveform[1]            # left track
        self.R = waveform[0]            # right track
        self.rate = rate                # sample rate
        self.filetype = filename[-4:]   # file extension
            #### Parameters Attributes ####
        self.chs = params[0]            # number of channels
        self.comptype = params[1]       # compression type
            #### Computed Attributes ####
        self.prd = 1/rate               # sample period
        self.nframes = len(self.L)      # num of frames in array
        self.length = self.nframes/self.rate
        self.time = np.arange(0,self.nframes)/self.rate
            #### Other Attributes ####

    def crop_silence (self,attrs=[],N=1024,bnd=0.1,overwrite=True):
        """
        Remove dead noise at the end of audio file
        --------------------------------
        attrs (list) : List of attribute strings to operate on
        N (int) : Number of Indicies incirment by
        bnd (float) : minimum amplitude threshold value (0.1 by default) 
        overwrite (bool) : If True, (default), overwrites exisiting attribute value
        --------------------------------
        Returns an array of arrays w/ eliminated ead noise
        """
        outputs = np.array([])                      # array to hold all outputs
        for attr in attrs:                          # attr to crop
            try:                                    # attempt
                data = self.__getattribute__(attr)  # isolate attr
                data = np.reshape(data,(-1,N))      # reshape to (m x N)
                while True:                             # indefinite loop
                    max1 = np.max(np.abs(data[-1]))     # max of last row
                    max0 = np.max(np.abs(data[0]))      # max of 1st row
                    if max1 < bnd:                      # if amp too small
                        data = np.delete(data,-1,0)     # delete last row
                    #elif max0 < bnd:                    # else if,
                    #    data = np.delete(data,0,0)      # delete last row
                    else:                               # otherwise
                        break                           # leave loop
                data = data.flatten()               # flatten array
                if overwrite == True:               # overwritre?
                        setattr(self,attr,data)     # reset attr value
                outputs = np.append(outputs,data)   # add output to array
            except:                                 # if failure,
                print("\n\tERROR! - Cannot Crop Attribite:",attr)
                attrs = attrs.remove(attr)          # remove attr from array
        return outputs.reshape(len(attrs),-1)       # return reshaped matrix

    def divisible_by_N (self,N=1024):
        """
        Make all time dependent array attribute a length divisible by N 
        --------------------------------
        N (int) : Number of columns in reshaped array
        --------------------------------
        Returns None, Resets all time dependent attributes
        """
        rem = np.mod(self.nframes,N)                    # remainer of frames
        pts = np.arange(0,self.nframes-rem)             # idxs to keep
        setattr(self,'nframes',self.nframes-rem)        # reset num frames
        setattr(self,'length',self.nframes/self.rate)   # reset length of file
        attrs = ['time','L','R']                        # attrs to reshape
        for attr in attrs:                              # for each 
            data = self.__getattribute__(attr)          # isolate data
            try:                                    # attempt
                data = data[pts]                    # slice attr
            except IndexError:                      # if failure
                pass                                # keep data as is
            setattr(self,attr,data)                 # reset attr array

    def Fast_Fourier_Transform (self,attrs=[]):
        """
        Compute Discrete Fast Fourier Transform of object
        --------------------------------
        attrs (list) : List of attribute strings to operate on
        --------------------------------
        Returns array of power spectrum for attribute array
        """
        outputs = np.array([])                      # array to hold outputs
        for attr in attrs:                          # for each attr
            try:                                    # attempt
                data = self.__getattribute__(attr)  # isolate atrribute
                fftdata = fftpack.fft(data)         # compute FFT           
                power = np.absolute(fftdata)**2     # power spectrum
                name = attr + '_FFT'                # create name
                setattr(self,name,power)            # set as new attrribute      
                outputs = np.append(outputs,power)  # add to output array
            except:                                 # failure
                print("\n\tERROR! - Cannot take FFT of Attribute:",attr)        
                attrs = attrs.remove(attr)          # remove attr from array
        fspace = self.Frequency_Space(len(power))
        return fspace,outputs.reshape(len(attrs),-1)   # return reshaped matrix 
 
    def Frequency_Space (self,length):
        """ Compute x-axis for Frequency Space """
        fspace = fftpack.fftfreq(length,self.prd)   # create f space bins
        setattr(self,'freq_space',fspace)           # set attribute
        return fspace                               # return array

    def hanning_window (self,N=1024,M=1):
        """
        Apply Hanning Window to attribute, every N idxs 
        NOTE: Arrays must be rectangular to (N x m) in order for arry to work with attributes
        --------------------------------
        N (int) : Number of Indicies to apply window too
        M (int) : Number of time to repeat the window
        --------------------------------
        Returns Hanning Taper Window attribute and Hanning Window Taper as (Nx1) array
        """ 
        hann = np.array([])             # array to hold hann window
        for I in range (int(M)):        # M times
            set = signal.hanning(N)     # hanning window
            hann = np.append(hann,set)  # add to arr
        setattr(self,'hanning',hann)    # set attribute
        return hann                     # return the window taper as array

    def normalize (self,attrs=[],N=1,overwrite=True):
        """
        Normalize and attribute to N
        --------------------------------
        attrs (list) : List of attribute strings to operate on
        N (float) : Numerical value to normalize array to (default=1)
        overwrite (bool) : If True, (default), overwrites exisiting attribute value
        --------------------------------
        Returns an array of arrays normalized to value N
        """
        output = np.array([])                       # array to hold outputs
        for attr in attrs:                          # each attr to normalize
            try:                                    # attempt
                data = self.__getattribute__(attr)  # isolate attr
                data = (data/np.max(data))*N        # divide by max
                if overwrite == True:               # overwrite?
                    setattr(self,attr,data)         # reset attr value
                output = np.append(output,data)     # add to output array
            except:                                 # failure
                print("\n\tERROR! - Cannot Normalize Attribite:",attr)
                attrs = attrs.remove(attr)          # remove attr from array
        return output.reshape(len(attrs),-1)        # return reshaped matrix
      
    def slicebyidx (self,attrs=[],slice=[],overwrite=True):
        """
        Slice and overwrite attribute by index
        --------------------------------
        attrs (list) : List of attribute strings to operate on
        slice (list) : List of values to slice array 
        overwrite (bool) : If True, (default), overwrites exisiting attribute value
        --------------------------------
        Returns an array of arrays normalized to value N
        """
        outputs = np.array([])                      # array to hold outputs
        for attr in attrs:                          # attr to slice
            try:                                    # attempt
                data = self.__getattribute__(attr)  # isolate attr
                data = data[slice]                  # slice data
                if overwrite == True:               # overwritre?
                    setattr(self,attr,data)         # reset attr value
                outputs = np.append(outputs,data)    # add output to array
            except:                                 # failure
                print("\n\tERROR! - Cannot Slice Attribite:",attr)
                attrs = attrs.remove(attr)          # remove attr from array
        return outputs.reshape(len(attrs),-1)       # return reshaped matrix

    def spectrogram(self,attr,npts=1024,ovlp=512):
        """ Use Scipy.Signal to comput spectrogram
        --------------------------------
        attrs (str) : Attribute Strings to operate on
        npts (int) : number of points in the spectrogram window
        ovlp (int) : number of points to overlap each window by
        --------------------------------
        Returns time array, frequency array and spectrogram matrix
        """
        data = self.__getattribute__(attr)
        f,t,Sxx = signal.spectrogram(data,fs=self.rate,
                            nperseg=npts,noverlap=ovlp)
        pts = np.where((f>=0)&(f<=4000))    # A to B Hz
        f = f[pts]                          # slice frequency space
        Sxx = Sxx[pts]                      # slice spectrogram
            
        name = attr+'_Sxx'          # attatch attributes
        setattr(self,name,Sxx)
        setattr(self,name+'_t',t)
        setattr(self,name+'_f',f)
        return f,t,Sxx              # return arrays

    def to_csv (self,name,attrs=[]):
        """
        Write attribute arrays pandas Dataframe and then to csv file
        --------------------------------
        name (str) : name of csv file to be written
        attrs (list) : list of attribute to write to csv 
            NOTE:  each attr must have same last dimension
        --------------------------------
        Returns Pandas Dataframe of inputted attributes
        """
        outputs = np.array([])          # array to hold all outputs
        labels = np.array([])           # array to hold col headers
        for attr in attrs:              # each attribute
            try:                                    # attempt
                data = self.__getattribute__(attr)  # data array
                outputs = np.append(outputs,data)   # add data to array
                labels = np.append(labels,attr)     # add header to list
            except:
                print("\n\tERROR! - Cannot Write Attribite:",attr)
                attrs = attrs.remove(attr)              # remove attr from array
        outputs = np.reshape(outputs,(len(attrs),-1))   # reshape array
        outputs = np.transpose(outputs)                 # tranpose
        frame = pd.DataFrame(data=outputs,columns=labels,
                             dtype=float)               # create pandas df
        frame.to_csv(name)                              # write frame ot csv
        return frame                                # return dataframe   

class wav_file ():
    """ Create Raw wavefile object """

    def __init__(self,root,file):
        """ Initialize Class Object """
        self.dirpath = root                     # intial storage path
        self.file = file                        # filename
        self.fftpath = self.FFT_path()          # path for FFT output
        self.spectpath = self.Spectro_path()    # path

    def FFT_path (self):
        """ Create path destination for frequency spectra """
        path = self.dirpath.replace('wav_audio','wav_frequency_spectra')
        return path

    def Spectro_path(self):
        """ Create path destination for frequency spectra """
        path = self.dirpath.replace('wav_audio','wav_spectrogram')
        return path

    def make_paths (self):
        """ Test if paths exist """
        paths = [self.fftpath,self.spectpath]
        for path in paths:          # for each path
            try:                    # try to move to it
                os.chdir(path)      # move to path
            except:                 # if failure
                os.makedirs(path)   # create the path

        #### FUNCTIONS DEFINTIONS ####

def read_wav (filename,chs=2):
    """
    Read Audio information from .wavfile using Scipy.io.wavfile() 
    --------------------------------
    filename (str) : Filename used to indentify which file to read in directory
    --------------------------------
    returns audio array and sample rate in Hz
    """
    rate,data = sciowav.read(filename)  # read wavefile
    if sciowav.WavFileWarning:
        pass
    data = np.transpose(data)           # transpose
    return rate,data                    # return the arrays

def read_directory(dir):
    """
    Read all files in given directory path
    --------------------------------
    dir (str) : desired directory path
    --------------------------------
    returns two lists of all '.wav' files
    """
    wav_objs = []                           # list to '.wav' hold files objs
    for roots,dirs,files in os.walk(dir):   # all objects in the path
        for file in files:                  # files in list of files
            if file.endswith('.wav'):       # if '.wav' file
                wav = wav_file(roots,file)  # make wav_file instance
                wav_objs.append(wav)        # add to list of wav file objs
    return wav_objs                         # return lists  of files

        #### PLOTTING & VISUALIZATION FUNCTIONS #####

def Plot_Time (obj,attrs=[],save=False,show=False):
    """
    Produce Matplotlib Figure of data in time domain
    --------------------------------
    obj (class) : object to plot attributes of
    attrs (list) : list of attribute strings to plot data of
    save (bool) : indicates to progam to save figure to cwd (False by default)
    show (bool) : indicates to progam to show figure (False by default)
    --------------------------------
    Returns None
    """
        #### Initializations ####
    plt.figure(figsize=(20,8))          
    plt.title(obj.name,size=40,weight='bold')
    plt.xlabel("Time",size=20,weight='bold')
    plt.ylabel("Amplitude",size=20,weight='bold')

    for attr in attrs:
        try:
            data = obj.__getattribute__(attr)  # isolate attribute
            plt.plot(obj.time,data,label=str(attr))
        except:
            print("\n\tERROR! - Could not plot attribute:",attr)
        
    plt.hlines(0,obj.time[0],obj.time[-1],color='black')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if save == True:
        plt.savefig(obj.name+'.time.png')
    if show == True:
        plt.show()
    plt.close()

def Plot_Freq (obj,attrs=[],save=False,show=False):
    """
    Produce Matplotlib Figure of data in Frequency Domain
    Note: Must have executed FFT & Freq_Space methods before calling
    --------------------------------
    obj (class) : object to plot attributes of
    attrs (list) : list of attribute strings to plot data of
    save (bool) : indicates to progam to save figure to cwd (False by default)
    show (bool) : indicates to progam to show figure (False by default)
    --------------------------------
    Returns None
    """
        #### Initializations ####
    plt.figure(figsize=(20,8))          
    plt.title(obj.name,size=40,weight='bold')
    plt.xlabel("Frequency [Hz]",size=20,weight='bold')
    plt.ylabel("Amplitude",size=20,weight='bold')

    for attr in attrs:
        try:
            data = obj.__getattribute__(attr)  # isolate attribute
            plt.plot(obj.freq_space,data,label=str(attr))
        except:
            print("\n\tERROR! - Could not plot attribute:",attr)
        
    plt.hlines(0,obj.freq_space[0],obj.freq_space[-1],color='black')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if save == True:
        plt.savefig(obj.name+'.freq.png')
    if show == True:
        plt.show()
    plt.close()

def Plot_Spectrogram (obj,Sxx,t,f,save=False,show=False):
    """
    Produce Matplotlib Figure of data in Frequency and Time Domain
        Note: Must have executed Spectrogram
    --------------------------------
    obj (class) : object to plot spectrogram of
    Sxx (array) : 2D spectrogram matrix
    t (array) : time axis array for spectrogram
    f (array) : frequency axis array for spectrogram
    save (bool) : indicates to progam to save figure to cwd (False by default)
    show (bool) : indicates to progam to show figure (False by default)
    --------------------------------
    Returns None
    """
    plt.figure(figsize=(20,8))          
    plt.title(obj.name,size=40,weight='bold')
    plt.xlabel("Time",size=20,weight='bold')
    plt.ylabel("Frequency",size=20,weight='bold')

    plt.pcolormesh(t,f,Sxx)

    plt.grid()
    plt.tight_layout()
    if save == True:
        plt.savefig(obj.name+'.freq.png')
    if show == True:
        plt.show()
    plt.close()

  
