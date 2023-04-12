# 1. 시작하기

NFT 추천 모델을 Pytorch로 구현한 코드입니다. 이더리움 기반의 NFT 거래 데이터를 사용해 모델이 구축되었습니다.

### 1.1. 가상환경 설정

먼저, Conda가 설치되어 있어야 합니다. 그리고 가상환경을 설치해 줍니다.

```
conda create -n NFT python=3.10.9
conda activate NFT
pip install -r requirements.txt
```

`requirements.txt`에 정리된 주요 패키지는 다음과 같습니다.

```
numpy==1.24.2
pandas==1.5.3
torch==1.13.1+cu116
torch-geometric==2.2.0
torch-scatter==2.1.0+pt113cu116
torch-sparse==0.6.16+pt113cu116
```

### 1.2. 데이터 준비

실험에 사용된 5개 데이터셋은 [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) 에서 내려받을 수 있습니다. 모든 데이터는 전부 전처리가 완료된 상태라서 모델에 바로 입력될 수 있습니다.

```
mkdir dataset/collections
```
**모든 데이터는`./dataset/collections` 경로 안에 위치해야 합니다.**

### 1.3. 모델 훈련 예시

`main.py` 파일을 실행하면 모델을 구현할 수 있습니다. 예를 들어 모델을 azuki 데이터셋에 훈련시키고 싶으면, 다음 커맨드를 실행합니다.

```
python main.py --collection azuki --epoch 10
```

만약 여러 가지 세팅의 `main.py` 파일을 동시에 실행하고 싶으면  `run.sh` 파일을 실행하면 됩니다. 예를 들어 모델을 모든 데이터셋에 대해 훈련시키고 싶으면 다음 커맨드를 실행합니다.

```
bash run.sh
```
위 커맨드를 실행하면 모델 훈련이 시작되고, 동시에 모델 테스트 결과가 출력됩니다. 

# 2. 코드 설명

## 2.1. `dataset` 

- `collection`: 각 NFT 콜렉션에서 발생한 유저-아이템 상호작용(interaction)과 특성(features) 데이터를 포함합니다. 예를 들어, `train.npy`는 유저-아이템 interaction 데이터이고 `image_feat.npy`는 전처리된 아이템 특성 데이터입니다.

- `features_user`: 유저 특성의 원본 데이터입니다. 각 유저의 거래 횟수, 평균 거래가격, 평균 보유기간을 포함합니다.

- `features_item`: 아이템 특성의 원본 데이터입니다. 각 아이템의 이미지, 텍스트, 가격, 거래 특성을 포함합니다.

## 2.2. `saved` 

모델 훈련 과정에서 가장 좋은 validation 성능을 보여주는 모델이 저장됩니다. 

## 2.2. `Create_dataset.ipynb` 

모델에서 요구하는 형식으로 입력 데이터 파일을 생성합니다. 생성된 파일은  [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) 에서 내려받을 수 있습니다. 

- `adj_dict.npy`: 그래프 노드 연결관계
- `train.npy`, `val.npy`,  `test.npy`: 훈련, 검증, 테스트 데이터셋
- `user_feat.npy`: 전처리된 유저 특성
- `image_feat.npy`, `text_feat.npy`,  `price_feat.npy`,  `transaction_feat.npy`: 전처리된 아이템 특성

## 2.2. `DataLoad.py` 

기존 훈련 데이터셋인 `train.npy`를 가지고 negative sampling을 수행하여 최종 훈련 데이터를 만듭니다. "pop_num" 매개변수(parameter)를 통해 negative sample 개수를 조절할 수 있습니다.

## 2.2. `GraphGAT.py` 

 `Model.py` 에서 사용되는 어텐션(Attention) 기반 Graph Convolutional Networks(GCN) 구조입니다. 훈련 중 드롭아웃 비율을 "DROPOUT" 매개변수(parameter)를 통해 조절할 수 있습니다.

## 2.2. `Model.py` 

NFT 추천시스템 모델 구조입니다. 모델은 Multi-graph embedding, Attention, Multi-objectives 등으로 이루어져 있습니다. 

## 2.2. `main.py` 

모델을 사용하여 실험을 실행하는 코드입니다. 데이터와 모델을 로드하고, Optimizer를 설정하고, 훈련 및 평가를 수행합니다.

### 커맨드 입력

- `collection`: 모델 구축에 사용할 데이터셋 이름 (NFT 콜렉션 이름)
- `data_path`: 데이터셋 저장 경로
- `l_r`: 모델 학습율
- `weight_decay`: 모델 최적화 과정에서 사용할 가중치 감쇠 (과적합 방지 목적)
- `batch_size`: 모델 훈련 배치 사이즈
- `dim_latent`: 최종 그래프 노드 임베딩 사이즈
- `num_epoch`: 훈련 epoch 개수
- `reg_parm`: 모델 훈련 시 정규화 강도 (과적합 방지 목적)
- `neg_sample`: 모델 훈련 negative sample 개수 
- `attention_dropout`: 모델의 그래프 어텐션 레이어에서 이웃 노드를 취합할 때 적용할 드랍아웃 비율
- `seed`: 랜덤시드 (실험 재현 목적)

### 커맨스 예시

```
python main.py --collection azuki --weight_decay 0 --batch_size 2048 --num_epoch 200 --attention_dropout 0.2 --seed 2023
```

# 3. (optional) 비교모델 실험

우리 모델의 성능을 기존 추천 모델들과 비교하기 위한 실험입니다. 추천시스템 라이브러리인 RecBole을 사용해 실험했습니다. 

### 3.1. 가상환경 설정

먼저, Conda가 설치되어 있어야 합니다. 그리고 가상환경을 설치해 줍니다:

```
conda create -n RecBole python=3.7.12
conda activate RecBole
pip install -r requirements.txt
```

`requirements.txt`에 정리된 주요 패키지는 다음과 같습니다:

```
pandas==1.3.5
hyperopt==0.2.7
ray==2.3.1
recbole==1.1.1
torch==1.12.1
```

### 3.2. 데이터 준비

실험에 사용된 5개 데이터셋은 [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) 에서 내려받을 수 있습니다. 모든 데이터는 전부 전처리가 완료된 상태라서 모델에 바로 입력될 수 있습니다.

```
mkdir dataset/collections
```

**모든 데이터는`./dataset/collections` 경로 안에 위치해야 합니다.**

### 3.3. 모델 훈련 예시

`main.py` 파일을 실행하면 모델을 구현할 수 있습니다. 예를 들어 모델을 azuki 데이터셋에 훈련시키고 싶으면, 다음 커맨드를 실행합니다.

```
python main.py --collection azuki 
```

만약 여러 가지 세팅의 `main.py` 파일을 동시에 실행하고 싶으면  `run.sh` 파일을 실행하면 됩니다. 예를 들어 모델을 모든 데이터셋에 대해 훈련시키고 싶으면 다음 커맨드를 실행합니다.

```
bash run.sh
```

위 커맨드를 실행하면 모델 훈련이 시작되고, 동시에 모델 테스트 결과가 출력됩니다. 

# Acknowlegement

Zhulin Tao, Yinwei Wei, Xiang Wang, Xiangnan He, Xianglin Huang, Tat-Seng Chua: MGAT: Multimodal Graph Attention Network for Recommendation. Inf. Process. Manag. 57(5): 102277 (2020)

https://github.com/zltao/MGAT

Zhao, W. X., Mu, S., Hou, Y., Lin, Z., Chen, Y., Pan, X., ... & Wen, J. R. (2021, October). Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 4653-4664).

https://github.com/RUCAIBox/RecBole
