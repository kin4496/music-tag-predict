import re
import sentencepiece as spm
import os
import numpy as np
import pandas as pd
import logging

RAW_DATA_DIR="../input/raw_data"
PROCESSED_DATA_DIR="../input/processed"
VOCAB_DIR = os.path.join(PROCESSED_DATA_DIR,'vocab')

# 학습에 사용될 파일 
TRAIN_FILE = 'train.json'

# 테스트에 사용될 파일 
DEV_FILE = 'data.json'

def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
logger = get_logger()

# 문장의 특수기호 제거 함수
def remove_special_characters(sentence, lower=True):

    p = re.compile('\([^)]*\)')
    sentence = p.sub(' ', sentence) # 패턴 객체로 sentence () 안의 문자열을 공백문자로 치환한다.

    p = re.compile('\<[^)]*\>')
    sentence = p.sub(' ', sentence) # 패턴 객체로 sentence 내의 <> 안의 문자열을 공백문자로 치환한다.

    p = re.compile('\[[^\]]*\]') 
    sentence = p.sub(' ', sentence) # 패턴 객체로 sentence 내의 [] 안의 문자열을 공백문자로 치환한다.

    # 특수기호를 나열한 패턴 문자열을 컴파일하여 패턴 객체를 얻는다.
    p = re.compile('[\!@#$%\^&\*\(\)\-\=\[\]\{\}\.,/\?~\+\'"|_:;><`┃]')    
    sentence = p.sub(' ', sentence) # 패턴 객체로 sentence 내의 특수기호를 공백문자로 치환한다.

    sentence = ' '.join(sentence.split()) # sentence 내의 두개 이상 연속된 빈공백들을 하나의 빈공백으로 만든다.

    if lower:
        sentence = sentence.lower()
    return sentence

def train_spm(txt_path,spm_path,vocab_size=32000,input_sentence_size=1000000):
    #vocab_size 사전 크기
    #input_sentence_size 개수만큼만 학습데이터로 사용된다.

    spm.SentencePieceTrainer.Train(
        f'--input={txt_path} --model_type=bpe '
        f'--model_prefix={spm_path} --vocab_size={vocab_size} '
        f'--input_sentence_size={input_sentence_size} '
        f'--shuffle_input_sentence=true'
    )
    
def get_dataframe(fname):
    df=pd.read_json(os.path.join(RAW_DATA_DIR,fname))
    
    text=[]
    for title,lyric in zip(df['title'],df['lyric']):
        if lyric != None:
            text.append(title+'\n'+lyric)
        else:
            text.append(title)
    
    df['text']=text
    return df

def preprocess():

    # PROCESSED_DATA_DIR과 VOCAB_DIR를 생성한다.
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(VOCAB_DIR, exist_ok=True)

    #데이터 프레임 불러오기
    train_df=get_dataframe(TRAIN_FILE)
    dev_df=get_dataframe(DEV_FILE)
    
    #dev_df에 topic,mood,situation column 추가하기
    labels=['topic','mood','emotion','situation']
    size=dev_df['title'].count()
    tempArr=[-1 for i in range(0,size)]
    for label in labels:
        if label not in dev_df.columns:
            dev_df[label]=tempArr
    

    # text 칼럼에 특수기호를 제거하는 함수를 적용한 결과를 반환한다.
    train_df['text'] = train_df['text'].map(remove_special_characters)
    dev_df['text'] = dev_df['text'].map(remove_special_characters)
    
    # 제목+가사를 text.txt 파일명으로 저장한다.
    with open(os.path.join(VOCAB_DIR, 'text.txt'), 'w', encoding='utf-8') as f:
        f.write(train_df['text'].str.cat(sep='\n'))
    

    # text.txt 파일로 sentencepiece 모델을 학습 시킨다. 
    # 학습이 완료되면 spm.model, spm.vocab 파일이 생성된다
    train_spm(txt_path=os.path.join(VOCAB_DIR, 'text.txt'), 
              spm_path=os.path.join(VOCAB_DIR, 'spm')) # spm 접두어

    # 센텐스피스 모델 학습이 완료되면 text.txt는 삭제
    os.remove(os.path.join(VOCAB_DIR, 'text.txt'))

    # 필요한 파일이 제대로 생성됐는지 확인
    for dirname, _, filenames in os.walk(VOCAB_DIR):
        for filename in filenames:
            logger.info(os.path.join(dirname, filename))

    logger.info('tokenizing title + lyric ...')
    
    # 센텐스피스 모델을 로드한다.
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(VOCAB_DIR, 'spm.model'))

    # text 칼럼의 제목 + 가사를 분절한 결과를 tokens 칼럼에 저장한다.
    train_df['tokens'] = train_df['text'].map(lambda x: " ".join(sp.EncodeAsPieces(x)) )
    dev_df['tokens'] = dev_df['text'].map(lambda x: " ".join(sp.EncodeAsPieces(x)) )
    
    #topic, mood, emotion, situation 컬럼의 값을 0,1,2,3... 같은 숫자로 바꾼다.
    train_df['topic']=train_df['topic'].astype('category').cat.codes
    train_df['mood']=train_df['mood'].astype('category').cat.codes
    train_df['emotion']=train_df['emotion'].astype('category').cat.codes
    train_df['situation']=train_df['situation'].astype('category').cat.codes
    
    #필요한 컬럼만 남기고 삭제
    columns = ['tokens','title',  'artist', 'topic', 'mood', 'emotion', 'situation']
    train_df=train_df[columns]
    
    #json 형태로 저장
    train_df.to_json(os.path.join(PROCESSED_DATA_DIR,'train.json'))
    dev_df.to_json(os.path.join(PROCESSED_DATA_DIR,'data.json'))
    
if __name__ == '__main__':
    preprocess()