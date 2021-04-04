function CNN = jCNN(imgs,label, suffix)
kfoldCount = 5;
fold    = cvpartition(label,'kfold',kfoldCount);
Afold   = zeros(kfoldCount,1); 
confmat = 0;
tic;
rand(10);
% ensuring that the same indexes are used in every feature extraction
kfold = [];
matFile = "features_all.mat";
if isfile(matFile)
   kfold = load(matFile, "kfold");
else
  for i = 1:kfoldCount
      train_idx  = fold.training(i);
      test_idx   = fold.test(i);
      key = strcat("idx_", num2str(i));
      kfold.(key).train = train_idx;
      kfold.(key).test = test_idx;
  end
  saveMatFile(matFile, "kfold", kfold);
end

saveMatFile(matFile, strcat("LBP_",suffix), getLBPFeatures(imgs.Files));
%saveMatFile(matFile, strcat("HOG_",suffix), getHOGFeatures(imgs.Files));
%saveMatFile(matFile, strcat("Mobilenetv2_",suffix), activations(mobilenetv2,imgs,'Logits','OutputAs', 'rows'));
saveMatFile(matFile, strcat("Densenet201_",suffix), activations(densenet201,imgs,'avg_pool','OutputAs', 'rows'));
%saveMatFile(matFile, strcat("Resnet18_",suffix), activations(resnet18,imgs,'pool5','OutputAs', 'rows'));

for i = 1:kfoldCount
  disp([num2str(i),' başladı.'])
  key = strcat("idx_", num2str(i));
  train_idx = kfold.kfold.(key).train;
  test_idx   = kfold.kfold.(key).test;
  
  xtrain = subset(imgs, train_idx);
  ytrain = label(train_idx);
  xtest = subset(imgs, test_idx);
  ytest = label(test_idx);
  
  featLBP = getFeatures("LBP", i, xtrain, xtest, ytrain, ytest);
  %featHOG = getFeatures("HOG", i, xtrain, xtest, ytrain, ytest);
  %featDensenet201 = getFeatures("densenet201", i, xtrain, xtest, ytrain, ytest);
  %featResnet18 = getFeatures("resnet18", i, xtrain, xtest, ytrain, ytest);
  %featMobilenetv2 = getFeatures("mobilenetv2", i, xtrain, xtest, ytrain, ytest);
  
  %featTrain = [featHOG.train(:,featHOG.Sf)];
  %featTest = [featHOG.test(:,featHOG.Sf)];
  %[Sf,Nf,curve] = binaryPSO(featTrain, featTest, ytrain, ytest, 10);
  featTrain = featLBP.train(:,featLBP.Sf);
  featTest = featLBP.test(:,featLBP.Sf);
  
  Pred = predictWithSVM(featTrain, ytrain, featTest);
  
  con        = confusionmat(ytest,Pred);
  confmat    = confmat + con; 
  Afold(i,1) = sum(diag(con)) / sum(con(:));
  disp([num2str(i),' bitti. Acc:', num2str(Afold(i,1))])
end

Acc  = mean(Afold);
time = toc;

CNN.acc = Acc; 
CNN.con = confmat;
CNN.t   = time;

fprintf('\n Classification Accuracy (CNN): %g %% \n ',100* Acc);
end

function Pred = predictWithSVM(featuresTrain, ytrain, featuresTest)
  t = templateSVM('SaveSupportVectors',true, 'KernelFunction','rbf', 'BoxConstraint', 100);
  classifier = fitcecoc(featuresTrain,ytrain, 'Learners', t, 'Coding', 'onevsall', 'ObservationsIn', 'rows');
  Pred = predict(classifier,featuresTest);
end

function [featuresTrain, featuresTest] = getFeaturesFromModel(net, layer, xtrain, xtest)
  featuresTrain = activations(net,xtrain,layer,'OutputAs', 'rows');
  featuresTest = activations(net,xtest,layer,'OutputAs', 'rows');
end 

function Features = getFeatures(method, i, xtrain, xtest, ytrain, ytest)
    matFile = "features_all.mat";
    Features = [];
    key = strcat(method, "_", num2str(i));
    if isfile(matFile)
        Features = load(matFile, key);
    end
    if ~isfield(Features, key)
        if method == "HOG"
            Features.train = getHOGFeatures(xtrain.Files);
            Features.test = getHOGFeatures(xtest.Files);
        elseif method == "LBP"
            Features.train = getLBPFeatures(xtrain.Files);
            Features.test = getLBPFeatures(xtest.Files);
        elseif method == "resnet18" 
            [Features.train, Features.test] = getFeaturesFromModel(resnet18, 'pool5', xtrain, xtest);
        elseif method == "mobilenetv2"
            [Features.train, Features.test] = getFeaturesFromModel(mobilenetv2, 'Logits', xtrain, xtest);
        elseif method == "densenet201"
            [Features.train, Features.test] = getFeaturesFromModel(densenet201, 'avg_pool', xtrain, xtest);
        elseif method == "vgg19"
            [Features.train, Features.test] = getFeaturesFromModel(vgg19, 'pool4', xtrain, xtest);
        end
        
        [Features.Sf,Features.Nf,Features.curve] = binaryPSO(Features.train, Features.test, ytrain, ytest);
 
        saveMatFile(matFile, key, Features);        
    else 
        Features = Features.(key);
    end
    

end

%PSO%
function [Sf,Nf,curve] = binaryPSO(xtrain, xtest, ytrain, ytest, max_Iter)
    N = 30; 
    c1       = 2; 
    c2       = 2;
    if ~exist('max_Iter','var')
      max_Iter = 20;
    end
    [Sf,Nf,curve] = jBPSO(xtrain, xtest, ytrain, ytest, N, max_Iter,c1,c2);
end

function HOGFeatures = getHOGFeatures(files)
  nrow = numel(files);
  HOGFeatures = zeros(nrow, 2304);
  for indx = 1:numel(files)
      filepath = files{indx};
      img = imread(filepath);
      img = im2double(img);
      %img = imresize(img,[256 256]);
      featureVector = extractHOGFeatures(img,'CellSize',[32 32]);
      HOGFeatures(indx, :) = featureVector;
  end
end


function LBPFeatures = getLBPFeatures(files)
  nrow = numel(files);
  LBPFeatures = zeros(nrow, 59);
  for indx = 1:numel(files)
      filepath = files{indx};
      img = imread(filepath);
      [rownum, colnum, numberOfColorChannels] = size(img);
      if numberOfColorChannels > 1
          %img = img(:, :, 2); % Take green channel.
          img = rgb2gray(img);
      end
      LBPFeatures(indx, :) =  extractLBPFeatures(img);
  end
end
