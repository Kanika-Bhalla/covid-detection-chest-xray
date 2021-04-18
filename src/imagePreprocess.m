function imagePreprocess(rootFolder, categories, outdir)

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
writeall(imds, outdir);

end