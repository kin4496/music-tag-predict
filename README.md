# music-tag-predict

소프트웨어 마에스트로에서 진행한 프로젝트로 뮤직 오토태깅을 주제로 한 프로젝트 소스코드입니다.
이번 프로젝트에서는 음원 추천에 기반이 되는 태그들을 자동으로 생성해주는 것을 목표로 합니다. 

```
├─code
│  └─preprocess.py #데이터 전처리 
│  └─tag_dataset.py #데이터 셋 정의
│  └─tag_model.py #모델 정의
│  └─train.py # 모델학습
│  └─inference.py # 모델 테스트
├─input
│  ├─processed
│  │  ├─music
│  │  │  ├─mel # mel-spec 저장되어있는 디렉토리
│  │  │  └─music 
│  │  └─vocab
│  └─raw_data
│  └─train.json
│  └─test.json
├─model #학습되 모델 파라미터를 저장하는 디렉토리
└─submission #모델 테스트 결과를 저장하는 디렉토리
```

## 모델 구조

![모델 구조](https://user-images.githubusercontent.com/48973279/197179360-21c6f54b-77f8-4a54-bb8b-9077b69456e9.png)

Text Encoder로 BERT를 사용하여 임베딩하였고 Sound Encoder로 음원파일을 Mel-spectogram 형태로 변환된 이미지를 CNN을 사용하여 feature vector를 뽑아내어 직렬로 연결해 Dense Layer에 넣어 분류문제로 태그를 생성했습니다. 

## 데이터 셋 구성하기

본 프로젝트에서는 노래의 가사와 음원파일을 Mel-spectogram으로 변환된 이미지를 사용하므로 데이터 셋을 다음과 같이 구성해서 사용할 수 있다.
노래의 제목과 가사 아티스트가 필요함으로 title,lyric,artist를 column으로 가지는 json 형태의 파일을 준비하고 해당 곡에 맞는 mel-spectogram 이미지를 mel 폴더에 저장한다.

## Getting Start

### 데이터 전처리하기

```
python preprocess.py
```

### 모델 학습하기

```
python train.py --batch_size 128 --nheads 8 --nlayers 8 --fold 0
```

### 모델 테스트 하기

```
python inference.py --batch_size 128 --nheads 8 --nlayers 8
```
