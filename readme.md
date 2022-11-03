## 설치
---

1. 파이썬 가상환경(3.7.13)을 생성한 뒤 repository를 clone합니다.
```
git clone https://github.com/hjk1996/pet_food_weight_estimator.git
```  
  
  
2. 모델의 dependency를 설치합니다.
```
pip install -r requirements.txt
```
  
  
## 모델 훈련
---

1. 수집한 이미지에 대한 메타 데이터를 담고 있는 image_meta_data.csv 파일을 생성합니다.  
image_meta_data.csv는 다음과 같은 column을 가지고 있습니다.  


|칼럼 이름|내용|예시|
|------|---|---|
|bowl_type|사료 용기 종류 (0부터 시작)|0|
|food_type|사료 종류 (0부터 시작)|3|
|gram|사료 무게|20|
|image_name|이미지 이름|image1.jpg|
  
  
2. 훈련 및 검증에 사용할 이미지 담고 있는 images 폴더와   
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


3. 명령어를 입력해 훈련 및 평가를 실시합니다.

```
# example
python train.py --epoch 500 --batch_size 16 --weights ./model_weights/best.pt
```


  훈련에서 설정할 수 있는 파라미터는 다음과 같습니다.


|파라미터|설명|기본값|
|------|---|---|
|epoch|훈련 epoch 수|1000|
|batch_size|데이터셋 batch 크기|32|
|hidden_size|FC layer 히든 레이어 유닛 수|768|
|classification|사료 종류 학습 여부, False로 설정하면 사료 중량만 학습하고 사료 종류는 학습하지 않음|True|
|n_classes|사료 종류 수, 사료 종류 분류 학습할 경우 입력 필수|21|
|test_size|전체 데이터셋에서 평가 데이터셋 비중|0.2|
|weights|모델 가중치 경로, 이전에 학습시켜 놓은 가중치가 있을 경우 사용|None|


