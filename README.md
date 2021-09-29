# Covid Detection Using Chest X-Ray Images

[Download original image dataset](https://drive.google.com/file/d/13jZ3cJLCNJoQ-YuoZwB-z3I6NKEiXloq/view?usp=sharing)

[Download preprocessed image dataset](https://drive.google.com/file/d/1PFK_uXuwo8mBOjj82nw6W4N-9ss0N45p/view?usp=sharing)

[Download extracted features mat file](https://drive.google.com/file/d/1AVN66IRWoiiemjmnTSOTG6HhNRq0bnUx/view?usp=sharing)

**Enviroment**
- Matlab 2020a

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

**Proposed Model**

<img width="666" alt="Proposed Model" src="https://user-images.githubusercontent.com/1481904/135340425-730238bb-8f24-4dbc-8cf2-d05938a82b77.png">
