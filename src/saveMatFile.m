function saveMatFile(matFile, key, val)
   a.(key) = val;
        
  if isfile(matFile)
     save(matFile, '-struct','a', key, '-append')
  else
     save(matFile, '-struct','a', key)
  end