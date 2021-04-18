% Read and pre-process images
function Iout = readAndPreprocessDensenet201(filename)

I = imread(filename);

[rows, columns, numberOfColorChannels] = size(I);
if rows ~= 224 || columns ~= 224
    Iout = imresize(I, [224 224]);
end

end