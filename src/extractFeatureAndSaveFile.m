function extractFeatureAndSaveFile(imgs, label, suffix, matFile)

saveMatFile(matFile, strcat("LBP_",suffix), getLBPFeatures(imgs.Files));
saveMatFile(matFile, strcat("Densenet201_",suffix), activations(densenet201,imgs,'avg_pool','OutputAs', 'rows'));

%saveMatFile(matFile, strcat("HOG_",suffix), getHOGFeatures(imgs.Files));
%saveMatFile(matFile, strcat("Mobilenetv2_",suffix), activations(mobilenetv2,imgs,'Logits','OutputAs', 'rows'));
%saveMatFile(matFile, strcat("Resnet18_",suffix), activations(resnet18,imgs,'pool5','OutputAs', 'rows'));

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
