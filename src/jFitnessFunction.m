% Notation: This fitness function is for demonstration 

function cost = jFitnessFunction(feat, label, X)
    if sum(X == 1) == 0
      cost = inf;
    else
        cost = jwrapperSVM(feat(:, X == 1), label);
    end
end

function error = jwrapperSVM(feat, label)
  acc = svmClassifier(feat, label, 3, 0);
  error = 1 - acc;
end

