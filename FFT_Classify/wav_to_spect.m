% ================
% Landon Buell
% Kevin Short
% Convert .aif to .wav
% 5 Novemeber 2019 
% ================


% clear workspace
clearvars;
clc;
%print("Program Running...")

%%%% Establish All directory Paths %%%%
rootdir = pwd;
readdir = 'C:\Users\Landon\Documents\wav_audio';

try                         % attempt to change dir
    chdir(outdir)           % change to path
catch                       % if failure, 
    mkdir(outdir)           % create the dir
end                     

chdir(readdir);                 % change to reading directory
files = dir('**\*.wav');        % all files in subfolder
strings = ["Violins","Violas","Violoncellos"];


try
    files(files == '.') = [];       % elminate '.'
    files(files == '..') = [];      % elminate '..'
catch
    % do nothing
end 

for i  = 1:length(files)
   
    filename = files(i).name;           % isolate filename
    folder = files(i).folder;           % isolate folder
    
    
    
end
    