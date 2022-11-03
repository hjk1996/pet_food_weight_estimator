# 설치
---

1. 파이썬 가상환경(3.7.13)을 생성한 뒤 repository를 clone합니다.
```
git clone https://github.com/hjk1996/pet_food_weight_estimator.git
```


2. 모델의 dependency를 설치합니다.
```
pip install -r requirements.txt
```


# 모델 훈련

0. 수집한 이미지에 대한 메타 데이터를 담고 있는 image_meta_data.csv 파일을 생성합니다.  
image_meta_data.csv는 다음과 같은 column을 가지고 있습니다.  

|제목|내용|설명|
|------|---|---|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|

1. 모델을 훈련하기 위해서는 데이터를 먼저 세팅해야 합니다.  
훈련 및 검증에 사용할 이미지 담고 있는 images 폴더와   
개별 이미지에 대한 정보를 담고 있는 image_meta_data.csv 파일을 data 폴더에 배치합니다.
```
root
│   
│   
└───data
    │   
    │   
    └───images
    │   │   
    │   │   
    │   └───image1.jpg
    │   └───image2.jpg
    │       ...
    │
    └───image_meta_data.csv
```
