function extractFeatureAndSaveFile(preprocessOutDir, categories, matFile)

imds = imageDatastore(fullfile(preprocessOutDir, categories), 'LabelSource', 'foldernames');
LBP.Extracted_Features = getLBPFeatures(imds.Files);
LBP.Labels = imds.Labels;
saveMatFile(matFile, "LBP", LBP);

imds = imageDatastore(fullfile(preprocessOutDir, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename)readAndPreprocessDensenet201(filename);
Densenet201.Extracted_Features = activations(densenet201, imds, 'avg_pool', 'OutputAs', 'rows');
Densenet201.Labels = imds.Labels;
saveMatFile(matFile, "Densenet201", Densenet201);

%saveMatFile(matFile, strcat("HOG_",suffix), getHOGFeatures(imgs.Files));
%saveMatFile(matFile, strcat("Mobilenetv2_",suffix), activations(mobilenetv2,imgs,'Logits','OutputAs', 'rows'));
%saveMatFile(matFile, strcat("Resnet18_","Extracted_Features"), activations(resnet18,imgs,'pool5','OutputAs', 'rows'));

end<

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
