import torch # 파이토치 패키지 임포트
from torch.utils.data import Dataset # Dataset 클래스 임포트
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import re
import os
import re


class TagDataset(Dataset):
    """
    데이터 셋에서 학습에 적합한 형태로 변환해서 반환
    """

    def __init__(self,df_data,mel_spec_path,token2id,tokens_max_len,type_vocab_size):
        """
        매개변수
        df_data : 노래 제목,가사,장르,가수가 저장되어있는 데이터 프레임
        mel_spec_path : 음원 멜-스펙트럼 파일 경로
        token2id : token을 token_id로 변환하기 위한 맵핑 정보를 가진 딕셔너리
        tokens_max_len : tokens의 최대 길이. 가사+제목의 tokens가 이 이상이면 버림
        type_vocab_size : 타입 사전의 크기
        """
        self.titles=df_data['title'].values
        self.artists=df_data['artist'].values
        self.transform=ToTensor()
        self.tokens = df_data['tokens'].values
        self.mel_spec_path = mel_spec_path
        self.tokens_max_len=tokens_max_len
        self.labels=df_data[['topic','mood','emotion','situation']].values
        self.token2id=token2id
        self.p=re.compile('▁[^▁]+') # ▁기호를 기준으로 나누기 위한 컴파일된 정규식
        self.type_vocab_size = type_vocab_size
    
    def __getitem__(self, idx):
        """
        데이터셋에서 idx에 대응되는 샘플을 변환하여 반환        
        """
        if idx >= len(self):
            raise StopIteration
        
        # idx에 해당하는 제목+가사 가져오기
        tokens=self.tokens[idx]
        if not isinstance(tokens,str):
            tokens = ''
        
        #제목+가사를 _기호로 분리하여 파이썬 리스트로 저장
        tokens=self.p.findall(tokens)

        # _기호 별 토큰타입 인덱스 부여
        token_types = [type_id for type_id,word in enumerate(tokens) for _ in word.split()]
        tokens = " ".join(tokens) # _기호로 분리되기 전의 원래의 tokens으로 되돌림

        token_ids = [self.token2id[tok] if tok in self.token2id else 0 for tok in tokens.split()]

        #token_ids의 길이가 max_len보다 길면 잘라서 버림
        if len(token_ids) > self.tokens_max_len:
            token_ids=token_ids[:self.tokens_max_len]
            token_types=token_types[:self.tokens_max_len]
        
        #token_ids의 길이가 max_len보다 짧으면 PAD(=0)으로 채움
        #token_ids 중 값이 있는 곳은 1, 그 외에는 0으로 채운 token_mask 생성
        #token_typesd에 max_len보다 짧은 만큼 PAD(=0) 채움
        token_mask = [1] * len(token_ids)
        token_pad = [0] * (self.tokens_max_len - len(token_ids))
        token_ids += token_pad
        token_mask += token_pad
        token_types += token_pad 

        # 넘파이(numpy)나 파이썬 자료형을 파이토치의 자료형으로 변환
        token_ids = torch.LongTensor(token_ids)
        token_mask = torch.LongTensor(token_mask)
        token_types = torch.LongTensor(token_types)

        # token_types의 타입 인덱스의 숫자,크기가 type_vocab_size 보다 작도록 바꿈
        token_types[token_types >=self.type_vocab_size] = self.type_vocab_size-1

        #음원의 멜-스펙트럼 이미지 가져오기
        title=self.titles[idx]
        artist=self.artists[idx]
        fname=self.getMelImageFname(title,artist)
        sound_feat=self.getMelImage(os.path.join(self.mel_spec_path,fname))
        
        #이미지가 없다면 0으로 채워진 텐서 반환
        if sound_feat==None:
            sound_feat=torch.zeros((3,200,200))

        # 주제/분위기/감정/상황 라벨 준비
        label = self.labels[idx]
        label = torch.LongTensor(label)

        # 제목+가사 텍스트 데이터, 음원 입력, 라벨을 반환한다.
        return token_ids, token_mask, token_types, sound_feat, label

    def __len__(self):
        """
          tokens의 개수를 반환한다. 즉, 상품명 문장의 개수를 반환한다.
        """
        return len(self.tokens)

    def getMelImageFname(self,title,artist):
        
        if artist==None or len(artist)==0:
            artist=''
        else:
            artist=artist[0]
            
        p=re.compile('[\|:?><*]')
        title=p.sub('_',title)
        artist=p.sub('_',artist)
        
        if artist=='':
            text=title
        else:
            text=f"{artist} {title}"
        
        text+='.jpg'
        return text

    def getMelImage(self,path,size=(200,200)):
        
        if os.path.exists(path):
            img=Image.open(path)
            img=img.resize(size)
            img.convert("RGB")
            img=self.transform(img)
        else:
            img=None
        return img