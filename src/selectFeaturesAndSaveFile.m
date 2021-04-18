function [Sf,Nf,curve] = selectFeaturesAndSaveFile(matFile)
    N = 30; 
    c1       = 2; 
    c2       = 2;
    if ~exist('max_Iter','var')
      max_Iter = 20;
    end
    
    data = load(matFile);
    [Sf,Nf,curve] = jBPSO(data.LBP.Extracted_Features, data.LBP.Labels, N, max_Iter,c1,c2);
    data.LBP.Sf = Sf;
    data.LBP.Nf = Nf;
    data.LBP.curve = curve;
    saveMatFile(matFile, "LBP", data.LBP);
    
    [Sf,Nf,curve] = jBPSO(data.Densenet201.Extracted_Features, data.Densenet201.Labels, N, max_Iter,c1,c2);
    data.Densenet201.Sf = Sf;
    data.Densenet201.Nf = Nf;
    data.Densenet201.curve = curve;
    saveMatFile(matFile, "Densenet201", data.Densenet201);
end