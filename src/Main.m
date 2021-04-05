%% (1) Convolutional Neural Network with one convolutional layer
clc, clear
% Benchmark dataset
suffix = "stretchlim";
rootFolder = fullfile(strcat('./data_', suffix, '/'));
categories = {'covid', 'normal', 'pneumo'};

%imds olusturma
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

% Convolutional Neural Network
CNN = jCNN(imds, imds.Labels, suffix);

% Accuracy
accuray = CNN.acc;
% Confusion matrix
confmat = CNN.con;
