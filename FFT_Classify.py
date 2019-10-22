"""
Landon Buell
FFT & Neural Network
Read raw Audio files (v1)
8 Sept 2019
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
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
                attrs.remove(attr)                  # remove attr from array
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
                attrs.remove(attr)                  # remove attr from array
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
                attrs.remove(attr)                  # remove attr from array
        return outputs.reshape(len(attrs),-1)       # return reshaped matrix

    def spectrogram (self,attr,N=1024,hann=True):
        """
        Create spectrogram for specific attrtibute
        --------------------------------
        attrs (str) : Attribute strings to operate on
            NOTE: Attribute must be able to be reshaped to N x M
        N (int) : Number of columns in reshaped array (1024 by default)
        hann (bool) : If True (default), apply Hanning window taper to row
            NOTE: 'hanning_window' method must have been used if True
        --------------------------------
        return M x N matrix of spectrogram data. 
            Each row is a linear frequency value and each column is a time increment
        """
        Sxx = np.array([])                  # array to hold sptgm data
        
        data = self.__getattribute__(attr)  # isolate attr
        data = np.reshape(data,(-1,N))      # reshape to (m x N)

        f = self.Frequency_Space(N)         # legnth of freq space
        pts = np.where((f>=0)&(f<=5000))    # A to B Hz
        f = f[pts]                          # slice frequency space

        for I in range (len(data)):             # for each row of data
            row = data[I]                       # islate data set
            if hann == True:                    # Hanning window?
                row *= self.hanning             # apply window
            fftdata = fftpack.fft(row)          # fft of I-th row
            power = np.absolute(fftdata)**2     # power spectrum     
            power = power[pts]                  # slice to f-spectrum
            power = power/np.max(power)         # normalize array to 1
            Sxx = np.append(Sxx,power)      # add array to spectrogram

        Sxx = Sxx.reshape(len(f),-1)        # reshape array
        Sxx = Sxx.transpose()               # transpose 
        t = np.arange(0,len(Sxx[-1])+1)*(N*self.prd)

        name = attr+'_Sxx'
        setattr(self,name,Sxx)
        setattr(self,'t',t)
        setattr(self,'f',f)
        return f,t,Sxx

    def scipy_spectrogram(self,attr,N=1024):
        """ use Scipy.Signal to comput spectrogram """
        data = self.__getattribute__(attr)
        f,t,Sxx = signal.spectrogram(data,
                    fs=self.rate,nperseg=N)
        pts = np.where((f>=0)&(f<=4000))    # A to B Hz
        f = f[pts]                          # slice frequency space
        Sxx = Sxx[pts]

        name = attr+'_Sxx'
        setattr(self,name,Sxx)
        setattr(self,'t',t)
        setattr(self,'f',f)
        return f,t,Sxx

    def to_csv (name,attrs=[]):
        """
        Write attribute arrays to csv file
        --------------------------------
        name (str) : name of csv file to be written
        attrs (list) : list of attribute to write to csv 
            NOTE:  each attr must have same last dimension
        --------------------------------
        """
        pass

    
        
    

class wav_file ():

    def __init__(self,root,file):
        """ Initialize Class Object """
        self.dirpath = root
        self.file = file
        self.fftpath = self.dirpath.replace('wav_audio','wav_spectra')
        self.spectpath = self.dirpath.replace('wav_audio','spectrogram')

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

def Plot_Spectrogram (obj,attr,save=False,show=False):
    """
    Produce Matplotlib Figure of data in Frequency and Time Domain
        Note: Must have executed Spectrogram
    --------------------------------
    obj (class) : object to plot spectrogram of
    save (bool) : indicates to progam to save figure to cwd (False by default)
    show (bool) : indicates to progam to show figure (False by default)
    --------------------------------
    Returns None
    """
    plt.figure(figsize=(20,8))          
    plt.title(obj.name,size=40,weight='bold')
    plt.xlabel("Time",size=20,weight='bold')
    plt.ylabel("Frequency",size=20,weight='bold')

    data = obj.__getattribute__(attr)       # isolate data array

    #data = np.exp(data)
    #plt.pcolormesh(obj.t,np.arange(len(data)),data)        # spectrogram
    plt.pcolormesh(obj.t,obj.f,data)             # from scipy method

    plt.grid()
    plt.tight_layout()
    if save == True:
        plt.savefig(obj.name+'.freq.png')
    if show == True:
        plt.show()
    plt.close()

  
