%% 
clc, clear

categories = {'covid', 'normal', 'pneumo'};
inputDir = 'data_original';
preprocessOutDir = './data_imreducehaze';
matFile = "features.mat";
kFold = 5;

%image preprocessing(reduce haze and resize(299x299))
imagePreprocess(inputDir, categories, preprocessOutDir);

% extract features and save mat file
extractFeatureAndSaveFile(preprocessOutDir, categories, matFile);

% select features with PSO
selectFeaturesAndSaveFile(matFile)

%svm classification
data = load(matFile);
features = [data.LBP.Extracted_Features(:, data.LBP.Sf), data.Densenet201.Extracted_Features(:, data.Densenet201.Sf)];
[validationAccuracy, validationPredictions, validationScores]  = svmClassifier(features,data.LBP.Labels,kFold,1);