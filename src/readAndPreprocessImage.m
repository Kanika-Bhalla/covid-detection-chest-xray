% Read and pre-process images
function Iout = readAndPreprocessImage(filename)

I = imread(filename);

% Some images may be grayscale. Replicate the image 3 times to
% create an RGB image.
if ismatrix(I)
    I = cat(3,I,I,I);    
end

I = imreducehaze(I);

[rows, columns, numberOfColorChannels] = size(I);
if rows ~= 299 || columns ~= 299
    Iout = imresize(I, [299 299]);
end

end