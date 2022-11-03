## 1. 설치
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
  
|칼럼 이름|내용|예시|
|------|---|---|
|bowl_type|사료 용기 종류 (0부터 시작)|0|
|food_type|사료 종류 (0부터 시작)|3|
|gram|사료 무게|20|
|image_name|이미지 이름|image1.jpg|
  
1. 
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
