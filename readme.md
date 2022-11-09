## 설치 및 구동환경 설정
---
docker를 통해 구동환경을 설정할 것을 권장합니다.  

1. docker 이미지를 다운로드 합니다.
```
docker pull hjk1996/iitp:0.2
```  
  
2. repository를 클론합니다. 
```
git clone https://github.com/hjk1996/pet_food_weight_estimator.git
```
  
3. clone된 repository 경로와 container 내의 작업경로를 매핑한 뒤 내려 받은 이미지로 가상환경을 생성합니다.
```
# gpu 사용하지 않는 경우
docker run -it -v <repository>:/workspace/ hjk1996/iitp:0.1 /bin/bash
# gpu 사용하는 경우
docker run --gpus '"device=0"' -it -v <repository>:/workspace/ hjk1996/iitp:0.1 /bin/bash
```
   
## 학습
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
  
### 2. 학습 시작
명령어를 입력해 모델 학습을 실시합니다.  
```
# example
python train.py --epoch 500 --batch_size 16 --weights ./model_weights/best.pt
```
  
학습 과정에서 설정할 수 있는 파라미터는 다음과 같습니다.  
|파라미터|설명|기본값|
|------|---|---|
|epoch|훈련 epoch 수|1000|
|batch_size|데이터셋 batch 크기|32|
|resize|리사이즈 이후 이미지의 너비와 높이, 학습 단계에서는 학습 속도를 위해 사용하지 않고 미리 모델의 인풋에 맞게 리사이징 된 이미지를 사용하는 것을 권장함|None|
|hidden_size|FC 레이어 유닛 수|768|
|classification|사료 종류 학습 여부, False로 설정하면 사료 중량만 학습하고 사료 종류는 학습하지 않음|True|
|n_classes|사료 종류 수, 사료 종류 분류 학습할 경우 입력 필수|21|
|test_size|전체 데이터셋에서 평가 데이터셋 비중|0.2|
|weights|모델 가중치 경로, 이전에 학습시켜 놓은 가중치가 있을 경우 사용|None|
  
학습 결과는 results 폴더에 저장됩니다.
```
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
  
## 학습 결과 확인
---
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


## 추론
---
학습된 가중치를 이용해서 사료 이미지의 중량과 종류에 대해 추론할 수 있습니다.
  
```
#example
python inference.py --image_path ./data/images/image1.jpg --weights ./model_weights/best.pt
```
|파라미터|설명|기본값|
|------|---|---|
|image_path|이미지 파일 경로||
|weights|모델 가중치 경로||
|resize|해당 파라미터의 값을 입력하면 입력한 사이즈로 이미지가 리사이즈된 후 모델의 입력으로 전달됨|None|
|classification|모델의 사료 종류 분류 기능 지원 여부|True|
|n_classes|모델이 사료 종류 분류가 가능할 경우 모델이 분류할 수 있는 사료의 종류의 수|21|
|mapping_path|모델의 사료 종류 예측에 대한 output과 실제 사료의 이름을 매핑해주는 json 파일의 경로. 없으면 사료 종류에 대한 indice만 반환함.|None|
