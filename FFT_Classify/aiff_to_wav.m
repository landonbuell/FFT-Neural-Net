% ================
% Landon Buell
% Kevin Short
% Convert .aif to .wav
% 2 October 2019
% ================


% clear workspace
clearvars;
clc;
%print("Program Running...")

%%%% Establish All directory Paths %%%%
rootdir = pwd;
readdir = 'C:\Users\Landon\Documents\aiff_audio';
outdir = strrep(readdir,'aiff_audio','wav_audio');

try                         % attempt to change dir
    chdir(outdir)           % change to path
catch                       % if failure, 
    mkdir(outdir)           % create the dir
end                     

chdir(readdir);                 % change to reading directory
files = dir('**\*.aif');        % all files in subfolder

try
    files(files == '.') = [];       % elminate '.'
    files(files == '..') = [];      % elminate '..'
catch
    % do nothing
end 

for i = 1:length(files)                 % in each file:
    
    filename = files(i).name;           % isolate filename
    dir = files(i).folder;              % isolate file folder
    chdir(dir);                         % change to specific folder
    newdir = strrep(dir,'aiff_audio','wav_audio');
    
    try                                         % try to read audio data
        [data,rate] = audioread(filename);      % read audio data
    catch
        disp("Data for file could not be read")
        disp(files(i).name)
        continue
    end 
    
    % data is L & R racks, rate is sample rate
    %data = cast(data,'int32');             % convert arr data type
    
    try                             % attempt
        chdir(newdir);              % change to new directory
    catch                           % failure:
        mkdir(newdir);              % make the dir path
    end
    
    outname = strrep(filename,'.aif','.wav');
    audiowrite(outname,data,rate);
    
end 
disp("Program Complete")