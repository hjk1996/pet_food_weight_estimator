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
### 1. 데이터 세팅
수집한 이미지에 대한 메타 데이터를 담고 있는 image_meta_data.csv 파일을 생성합니다.  
image_meta_data.csv는 다음과 같은 column을 가지고 있습니다.  

|칼럼 이름|내용|예시|
|------|---|---|
|bowl_type|사료 용기 종류 (0부터 시작)|0|
|food_type|사료 종류 (0부터 시작)|3|
|gram|사료 무게|20|
|image_name|이미지 이름|image1.jpg|
  
훈련 및 검증에 사용할 이미지 담고 있는 images 폴더와   
개별 이미지에 대한 정보를 담고 있는 image_meta_data.csv 파일을 data 폴더에 배치합니다.
  ```
  #example
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

### 2. 학습

명령어를 입력해 모델 학습을 실시합니다.  
```
# example
python train.py --epoch 500 --batch_size 16 --weights ./model_weights/best.pt
```

훈련에서 설정할 수 있는 파라미터는 다음과 같습니다.  
|파라미터|설명|기본값|
|------|---|---|
|epoch|훈련 epoch 수|1000|
|batch_size|데이터셋 batch 크기|32|
|hidden_size|FC 레이어 유닛 수|768|
|classification|사료 종류 학습 여부, False로 설정하면 사료 중량만 학습하고 사료 종류는 학습하지 않음|True|
|n_classes|사료 종류 수, 사료 종류 분류 학습할 경우 입력 필수|21|
|test_size|전체 데이터셋에서 평가 데이터셋 비중|0.2|
|weights|모델 가중치 경로, 이전에 학습시켜 놓은 가중치가 있을 경우 사용|None|

훈련 결과는 results 폴더에 저장됩니다.
  ```
  #example
  root
  │   
  │   
  └───results
      │   
      │   
      └───2022-11-03_10-37-07
          └───log (훈련 log가 저장되는 폴더)
          └───best.pt (최고 성능을 기록한 모델 가중치)
          └───last.pt (훈련 종료 시점 모델 가중치)
  ```
### 3. 결과 확인

tensorboard를 통해 학습 결과를 확인할 수 있습니다.  
다음 명령어를 실행한 뒤 제공하는 URL로 이동하거나 http://localhost:6006/ 로 이동해 학습 결과를 확인합니다.
```
#example
tensorboard --logdir=results/2022-11-03_10-37-07/log
```

|지표|설명|
|------|---|
|total_loss|Binary Cross Entropy(사료 종류 분류 오차) + MAE(중량 예측 오차)|
|mae|중량 예측 오차(평균 절대 오차)|
|acc|사료 종류 분류 정확도|


$MAE=1n∑i=1n|yi−y^i|$

$MAE=\frac{1}{n}\sum_{i=1}^{n}|x_i-y_i|$
