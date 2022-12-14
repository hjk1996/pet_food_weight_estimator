## 설치 및 구동환경 설정

---

docker를 통해 구동환경을 설정할 것을 권장합니다.

1. docker 이미지를 다운로드 합니다.

```
docker pull hjk1996/iitp:0.4
```

2. repository를 클론합니다.

```
git clone --recursive https://github.com/hjk1996/pet_food_weight_estimator.git
```

3. clone된 repository 경로와 container 내의 작업경로를 매핑한 뒤 내려 받은 이미지로 가상환경을 생성합니다.

```
# gpu 사용하지 않는 경우
docker run -it -p 6006:6006 -v <repository>:/workspace/ hjk1996/iitp:0.4 /bin/bash
# gpu 사용하는 경우
docker run --gpus '"device=0"' -it -p 6006:6006 -v <repository>:/workspace/ hjk1996/iitp:0.4 /bin/bash
```

## 학습

---

### 1. 설정

train_config.json을 통해 훈련에 필요한 변수들을 설정합니다.

| 칼럼 이름            | 내용                                                                                                                | 초기값                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| epoch                | 훈련 반복 횟수                                                                                                      | 500                        |
| batch_size           | 배치 사이즈                                                                                                         | 32                         |
| test_size            | 테스트 데이터 비중                                                                                                  | 0.2                        |
| model_name           | 사용할 backbone 모델 이름 (현재 swinv2 및 efficiientnet_v2 지원)                                                     | efficientnetv2_s   |
| num_classes          | 사료 종류수                                                                                                         | 21                         |
| on_memory            | 데이터셋을 모두 메모리에 올리고 학습할 지 여부. true로 설정시 학습 속도가 증가할 수 있으나 메모리가 부족할 수 있음. | false                      |
| image_folder_path    | 이미지가 저장된 폴더 경로                                                                                           | ./data/images              |
| image_meta_data_path | 이미지 메타 데이터 경로                                                                                             | ./data/image_meta_data.csv |
| weight_path          | 학습된 모델 가중치 경로. 설정시 해당 가중치를 이용해 학습을 진행함.                                                 | null                       |
| num_workers          | data loading시 사용하는 subprocess의 수. 0은 메인 프로세스 하나만 의미함.                                           | 8                          |
| cropper_weight_path  | object detection을 위해서 사용하는 yolov7의 가중치 경로                                                             | null                       |
| cropper_input_size   | yolov7의 입력으로 들어가는 이미지 크기                                                                              | null                       |
| cropper_output_size  | yolov7의 출력(crop된 이미지) 이미지의 크기                                                                          | null                       |

훈련과 검증을 위해서 모든 이미지가 저장되어 있는 images 폴더와  
개별 이미지에 대한 메타 정보를 담고 있는 image_meta_data.csv 파일이 필요합니다.

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
train과 관련된 설정이 저장된 json 파일의 경로를 파라미터로 전달합니다.
훈련 과정에서 설정할 수 있는 command line 파라미터는 다음과 같습니다.

|파라미터|설명|기본값|
| -------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------- |
|--config_path|훈련 설정 json 파일의 경로 |./train_config.json|
|--device|훈련에 사용할 장치. 장착된 gpu를 모두 사용하는 경우 'cuda', 특정 gpu를 사용하는 경우 'cuda:0' 또는 'cuda:0,1' 등, cpu를 사용할 경우 'cpu' |cuda|
|--test_mode|해당 파라미터를 전달할 경우 테스트 모드로 학습을 실행함. 모델과 훈련 과정이 정상적으로 진행되는 것을 빠르게 확인하고 싶을 떄 사용.||

아래는 예시 학습 코드입니다.

```
# example
python train.py --config_path <train_config file path> --device cuda:0
```

학습 결과는 train_confing.json에서 설정한 save_path로 지정한 경로에 저장됩니다.

```
<save_path>
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

| 지표       | 설명                                                              |
| ---------- | ----------------------------------------------------------------- |
| total_loss | Binary Cross Entropy(사료 종류 분류 오차) + RMSE (중량 예측 오차) |
| rmse       | 중량 예측 오차(평균 절대 오차)                                    |
| acc        | 사료 종류 분류 정확도                                             |

## 추론

---

학습된 가중치를 이용해서 사료 이미지의 중량과 종류에 대해 추론할 수 있습니다.
추론과 관련한 설정은 inference_config.json에서 수정할 수 있습니다.

```
#example
python inference.py --image_path ./data/images/image1.jpg --weights ./model_weights/best.pt
```

| 파라미터     | 설명                                                                                                                           | 기본값 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------ | ------ |
| img_path   | 이미지 파일 경로                                                                                                               |        |
| config_path      | 추론 설정 파일 경로                                                                                                              |   ./config_path.json     |
| resize       | 해당 파라미터의 값을 입력하면 입력한 사이즈로 이미지가 리사이즈된 후 모델의 입력으로 전달됨                                    | None   |
