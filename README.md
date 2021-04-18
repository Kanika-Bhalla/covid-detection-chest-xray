# covid-detection-chest-xray

[Download original image dataset](https://drive.google.com/file/d/13jZ3cJLCNJoQ-YuoZwB-z3I6NKEiXloq/view?usp=sharing)

[Download preprocessed image dataset](https://drive.google.com/file/d/1PFK_uXuwo8mBOjj82nw6W4N-9ss0N45p/view?usp=sharing)

[Download extracted features mat file](https://drive.google.com/file/d/1AVN66IRWoiiemjmnTSOTG6HhNRq0bnUx/view?usp=sharing)

**Running Steps:**

1) The original data directory should be as follows
```
data_original
  |--- covid
         |--- covid_img1.png
         |--- ....
  |--- normal
         |--- normal_img1.png
         |--- ....
  |--- pneumo
         |--- pneumo_img1.png
         |--- .....
```

2) Place the prepared image directory into the ```src``` directory
3) Run ```src/Main.m```  file
