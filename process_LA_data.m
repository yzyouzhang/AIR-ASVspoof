clear; close all; clc;

% This code is modified from the baseline system for ASVspoof 2019

% add required libraries to the path
addpath(genpath('LFCC'));
addpath(genpath('CQCC_v1.0'));

% set here the experiment to run (access and feature type)
access_type = 'LA';
feature_type = 'LFCC';

% set paths to the wave files and protocols
pathToASVspoof2019Data = 'D:\Programming\Python\Python\AIR-ASVspoof\DS_10283_3336';
pathToFeatures = horzcat('D:\Programming\Python\Python\AIR-ASVspoof\DS_10283_3336\anti-spoofing\ASVspoof2019\', access_type, '\Features\');

pathToDatabase = fullfile(pathToASVspoof2019Data, access_type);
trainProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_cm_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.train.trn.txt'));
devProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_cm_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.dev.trl.txt'))
evalProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_cm_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.eval.trl.txt'));

% read train protocol
trainfileID = fopen(trainProtocolFile);
trainprotocol = textscan(trainfileID, '%s%s%s%s%s');
fclose(trainfileID);
trainfilelist = trainprotocol{2};

% read dev protocol
devfileID = fopen(devProtocolFile);
devprotocol = textscan(devfileID, '%s%s%s%s%s');
fclose(devfileID);
devfilelist = devprotocol{2};

% read eval protocol
evalfileID = fopen(evalProtocolFile);
evalprotocol = textscan(evalfileID, '%s%s%s%s%s');
fclose(evalfileID);
evalfilelist = evalprotocol{2};


%% Feature extraction for training data

% extract features for training data and store them
disp('Extracting features for training data...');
trainFeatureCell = cell(length(trainfilelist), 3);
for i=1:length(trainfilelist)
    filePath = fullfile(pathToDatabase,['ASVspoof2019_' access_type '_train\flac'],[trainfilelist{i} '.flac']);
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    LFCC = [stat delta double_delta]';
    filename_LFCC = fullfile(pathToFeatures, 'train', horzcat('LFCC_', trainfilelist{i}, '.mat'))
    parsave(filename_LFCC, LFCC)
    LFCC = [];
end
disp('Done!');

%% Feature extraction for development data

% extract features for training data and store them
disp('Extracting features for development data...');
for i=1:length(devfilelist)
    filePath = fullfile(pathToDatabase,['ASVspoof2019_' access_type '_dev\flac'],[devfilelist{i} '.flac']);
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    LFCC = [stat delta double_delta]';
    filename_LFCC = fullfile(pathToFeatures, 'dev', horzcat('LFCC_', devfilelist{i}, '.mat'))
    parsave(filename_LFCC, LFCC)
    LFCC = [];
end
disp('Done!');

%% Feature extraction for evaluation data

% extract features for training data and store them
disp('Extracting features for evaluation data...');
for i=1:length(evalfilelist)
    filePath = fullfile(pathToDatabase,['ASVspoof2019_' access_type '_eval\flac'],[evalfilelist{i} '.flac']);
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    LFCC = [stat delta double_delta]';
    filename_LFCC = fullfile(pathToFeatures, 'eval', horzcat('LFCC_', evalfilelist{i}, '.mat'))
    parsave(filename_LFCC, LFCC)
    LFCC = [];
end
disp('Done!');


%% supplementary function
function parsave(fname, x)
    save(fname, 'x')
end
