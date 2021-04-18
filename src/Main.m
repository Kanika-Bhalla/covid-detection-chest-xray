%% 
clc, clear
suffix = "imreducehaze";
rootFolder = fullfile(strcat('./data_', suffix, '/'));
categories = {'covid', 'normal', 'pneumo'};

%imds olusturma
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

matFile = "features.mat";

% extract features and save mat file
extractFeatureAndSaveFile(imds, imds.Labels, suffix, matFile);

% select features with PSO
selectFeatureAndSaveFile(matFile)