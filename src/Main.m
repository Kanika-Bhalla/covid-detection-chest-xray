%---------------------------------------------------------------------%
%  Deep learning algorithm source codes demo version                  %
%---------------------------------------------------------------------%


%---Input--------------------------------------------------------------
% imgs      : feature vector (height x width x channel x instances)
% label     : label vector (instances x 1)
% kfold     : Number of cross-validation
% LR        : Learning rate
% nB        : Number of mini batch
% MaxEpochs : Maximum number of Epochs
% FC        : Number of fully connect layer (number of classes)
% nC        : Number of convolutional layer (up to 3)
% nF1       : Number of filter in first convolutional layer
% sF1       : Size of filter in first convolutional layer
% nF2       : Number of filter in second convolutional layer
% sF2       : Size of filter in second convolutional layer
% nF3       : Number of filter in third convolutional layer
% sF3       : Size of filter in third convolutional layer

%---Output-------------------------------------------------------------
% A struct that contains three results as follows:
% acc       : Overall accuracy
% con       : Confusion matrix
% t         : computational time (s)
%-----------------------------------------------------------------------


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



