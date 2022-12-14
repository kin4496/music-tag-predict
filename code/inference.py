import os
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['NUMEXPR_MAX_THREADS'] = '24'
import math
import glob
import torch
import tag_dataset
import tag_model
import time
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings(action='ignore')
import argparse

# gpu 옵션 설정하기
if torch.cuda.is_available():
    device=torch.device('cuda')
elif torch.backends.mps.is_available():
    device=torch.device('mps')
else:
    device=torch.device('cpu')

# 전처리 전 데이터가 저장된 디렉토리
RAW_DATA_DIR="../input/raw_data"

# 전처리된 데이터가 저장된 디렉터리
DB_DIR = '../input/processed'

# 토큰을 인덱스로 치환할 때 사용될 사전 파일이 저장된 디렉터리 
VOCAB_DIR = os.path.join(DB_DIR, 'vocab')

# 학습된 모델의 파라미터가 저장될 디렉터리
MODEL_DIR = '../model'

# 태그 예측결과가 저장될 디렉터리
SUBMISSION_DIR = '../submission'

# 미리 정의된 설정 값
class CFG:    
    batch_size=256 # 배치 사이즈
    num_workers=4 # 워커의 개수
    print_freq=100 # 결과 출력 빈도    
    warmup_steps=100 # lr을 서서히 증가시킬 step 수        
    hidden_size=512 # 은닉 크기
    dropout=0.2 # dropout 확률
    intermediate_size=256 # TRANSFORMER셀의 intermediate 크기
    nlayers=2 # BERT의 층수
    nheads=8 # BERT의 head 개수
    seq_len=256 # 토큰의 최대 길이
    n_t_cls = 0 # 주제 태그 개수
    n_m_cls = 0 # 분위기 태그 개수
    n_e_cls = 0 # 감정 태그 개수
    n_s_cls = 0 # 상황 태그 개수
    vocab_size = 32000 # 토큰의 유니크 인덱스 개수
    type_vocab_size = 2 # 타입의 유니크 인덱스 개수
    data_path = os.path.join(DB_DIR, 'data.json') # 전처리 돼 저장된 dev 데이터셋    
    mel_spec_path = os.path.join(DB_DIR, 'music/mel')

def main():
    # 명령행에서 받을 키워드 인자를 설정합니다.
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--data_dir", type=str, default='data.json')     
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)   
    parser.add_argument("--seq_len", type=int, default=CFG.seq_len)
    parser.add_argument("--nworkers", type=int, default=CFG.num_workers)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nlayers", type=int, default=CFG.nlayers)
    parser.add_argument("--nheads", type=int, default=CFG.nheads)
    parser.add_argument("--hidden_size", type=int, default=CFG.hidden_size)        
    args = parser.parse_args()
    print(args) 
    
    CFG.data_path=os.path.join(DB_DIR,args.data_dir)
    CFG.batch_size=args.batch_size    
    CFG.seed =  args.seed        
    CFG.nlayers =  args.nlayers    
    CFG.nheads =  args.nheads
    CFG.hidden_size =  args.hidden_size
    CFG.seq_len =  args.seq_len
    CFG.num_workers=args.nworkers
    
    # 전처리되기 전 데이터 읽어와 분류해야할 수를 가져온다.
    raw_train_df=pd.read_json(os.path.join(RAW_DATA_DIR,'train.json'))
    CFG.n_t_cls=len(raw_train_df['topic'].astype('category').cat.categories)
    CFG.n_m_cls=len(raw_train_df['mood'].astype('category').cat.categories)
    CFG.n_e_cls=len(raw_train_df['emotion'].astype('category').cat.categories)
    CFG.n_s_cls=len(raw_train_df['situation'].astype('category').cat.categories)
    
    # CFG 출력
    print(CFG.__dict__)    
    
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 함
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    if torch.cuda.is_available():    
        torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    
    # 전처리된 데이터를 읽어옵니다.
    print('loading ...')
    dev_df = pd.read_json(CFG.data_path)     
    mel_spec_path = CFG.mel_spec_path
    
    vocab = [line.split('\t')[0] for line in open(os.path.join(VOCAB_DIR, 'spm.vocab'), encoding='utf-8').readlines()]
    token2id = dict([(w, i) for i, w in enumerate(vocab)])    
    print('loading ... done')
        
    # 찾아진 모델 파일의 개수만큼 모델을 만들어서 파이썬 리스트에 추가함
    model_list = []
    # args.model_dir에 있는 확장자 .pt를 가지는 모든 모델 파일의 경로를 읽음
    model_path_list = glob.glob(os.path.join(args.model_dir, '*.pt'))
    # 모델 경로 개수만큼 모델을 생성하여 파이썬 리스트에 추가함
    for model_path in model_path_list:
        model = tag_model.TagClassifier(CFG)
        if model_path != "":
            print("=> loading checkpoint '{}'".format(model_path))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path,map_location=torch.device('cpu'))        
            state_dict = checkpoint['state_dict']                
            model.load_state_dict(state_dict, strict=True)  
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
        
        # GPU 사용가능한지 확인
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            print('Gpu 사용할 수 없습니다.')
            exit()
        
        # 모델의 파라미터를 GPU 메모리로 옮김
        model.to(device)

        # GPU가 2개 이상이면 데이터 패럴렐로
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        
        model_list.append(model)

    if len(model_list) == 0:
        print('Please check the model directory.')
        return
    
    # 모델의 파라미터 수를 출력합니다.
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('parameters: ', count_parameters(model_list[0]))    
    
    # 모델의 입력에 적합한 형태의 샘플을 가져오는 TagDataset의 인스턴스를 만듦
    dev_db = tag_dataset.TagDataset(dev_df, mel_spec_path, token2id, CFG.seq_len, 
                                       CFG.type_vocab_size)
    
    # 여러 개의 워커로 빠르게 배치(미니배치)를 생성하도록 DataLoader로 
    # TagDataset 인스턴스를 감싸 줌
    dev_loader = DataLoader(
        dev_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True)    
    
    # dev 데이터셋의 모든 노래에 대해 예측된 태그 인덱스를 반환
    pred_idx = inference(dev_loader, model_list)
    
    # dev 데이터셋의 노래별 예측된 태그를 붙이기
    tag_cols = ['topic', 'mood', 'emotion', 'situation'] 
    dev_df[tag_cols] = pred_idx
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    #숫자로 이루어진 태그를 알아보기 쉽도록 바꾸기
    label2topic=raw_train_df['topic'].astype('category').cat.categories.to_list()
    label2mood=raw_train_df['mood'].astype('category').cat.categories.to_list()
    label2emotion=raw_train_df['emotion'].astype('category').cat.categories.to_list()
    label2situation=raw_train_df['situation'].astype('category').cat.categories.to_list()
    
    dev_df['topic']=dev_df['topic'].map(lambda x: label2topic[x])
    dev_df['mood']=dev_df['mood'].map(lambda x: label2mood[x])
    dev_df['emotion']=dev_df['emotion'].map(lambda x: label2emotion[x])
    dev_df['situation']=dev_df['situation'].map(lambda x:label2situation[x])
    
    #제출 파일을 생성하여 저장
    song_cols=['title','artist']
    submission_path_json = os.path.join(SUBMISSION_DIR, 'test.json')
    submission_path_excel = os.path.join(SUBMISSION_DIR, 'test.xlsx')
    dev_df[song_cols+tag_cols].to_json(submission_path_json)
    dev_df[song_cols+tag_cols].to_excel(submission_path_excel)        
    print('done')

def inference(dev_loader, model_list):
    """
    dev 데이터셋의 모든 노래에 대해 여러 모델들의 예측한 결과를 앙상블 하여 정확도가 개선된
    카테고리 인덱스를 반환
    
    매개변수
    dev_loader: dev 데이터셋에서 배치(미니배치) 불러옴
    model_list: args.model_dir에서 불러온 모델 리스트 
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()    
    sent_count = AverageMeter()
    
    # 모딜 리스트의 모든 모델을 평가(evaluation) 모드로 작동하게 함
    for model in model_list:
        model.eval()

    start = end = time.time()
    
    # 배치별 예측한 주제/분위기/감정/상황의 인덱스를 리스트로 가짐
    pred_idx_list = []
    
    # dev_loader에서 반복해서 배치 데이터를 받음
    # TagDataset의 __getitem__() 함수의 반환 값과 동일한 변수 반환
    for step, (token_ids, token_mask, token_types, img_feat, _) in enumerate(dev_loader):
        # 데이터 로딩 시간 기록
        data_time.update(time.time() - end)
        
        # 배치 데이터의 위치를 CPU메모리에서 GPU메모리로 이동
        token_ids, token_mask, token_types, img_feat = (
            token_ids.to(device), token_mask.to(device), token_types.to(device), img_feat.to(device))
        
        batch_size = token_ids.size(0)
        
        # with문 내에서는 그래디언트 계산을 하지 않도록 함
        with torch.no_grad():
            pred_list = []
            # model 별 예측치를 pred_list에 추가합니다.
            for model in model_list:
                _, pred = model(token_ids, token_mask, token_types, img_feat)
                pred_list.append(pred)
            
            # 예측치 리스트를 앙상블 하여 하나의 예측치로 만듦
            pred = ensemble(pred_list)
            # 예측치에서 카테고리별 인덱스를 가져옴
            pred_idx = get_pred_idx(pred)
            # 현재 배치(미니배치)에서 얻어진 카테고리별 인덱스를 pred_idx_list에 추가
            pred_idx_list.append(pred_idx.cpu())
            
        # 소요시간 측정
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        if step % CFG.print_freq == 0 or step == (len(dev_loader)-1):
            print('TEST: {0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '                  
                  'sent/s {sent_s:.0f} '
                  .format(
                   step+1, len(dev_loader), batch_time=batch_time,                   
                   data_time=data_time,
                   remain=timeSince(start, float(step+1)/len(dev_loader)),
                   sent_s=sent_count.avg/batch_time.avg
                   ))
    
    # 배치별로 얻어진 태그 인덱스 리스트를 직렬연결하여 하나의 카테고리 인덱스로 변환
    pred_idx = torch.cat(pred_idx_list).numpy()
    return pred_idx

# 예측치의 각 카테고리 별로 가장 큰 값을 가지는 인덱스를 반환함
def get_pred_idx(pred):
    t_pred, m_pred, e_pred,s_pred = pred # 주제/분위기/감정/상황 예측치로 분리
    _, t_idx = t_pred.max(1) # 주제 중 가장 큰 값을 가지는 인덱스를 변수에 할당
    _, m_idx = m_pred.max(1) # 분위기 중 가장 큰 값을 가지는 인덱스를 변수에 할당
    _, e_idx = e_pred.max(1) # 감정 중 가장 큰 값을 가지는 인덱스를 변수에 할당
    _, s_idx = s_pred.max(1) # 상황 중 가장 큰 값을 가지는 인덱스를 변수에 할당
    
    # 주제/감정/분위기/상황 인덱스 반환
    pred_idx = torch.stack([t_idx, m_idx, e_idx, s_idx], 1)    
    return pred_idx


# 예측된 주제/분위기/감정/상황 결과들을 앙상블함
# 앙상블 방법으로 간단히 산술 평균을 사용
def ensemble(pred_list):
    t_pred, m_pred, e_pred, s_pred = 0, 0, 0, 0    
    for pred in pred_list:
        # softmax를 적용해 주제/분위기/감정/상황 각 태그별 모든 클래스의 합이 1이 되도록 정규화
        
        t_pred += torch.softmax(pred[0], 1)
        m_pred += torch.softmax(pred[1], 1)
        e_pred += torch.softmax(pred[2], 1)
        s_pred += torch.softmax(pred[3], 1)

    t_pred /= len(pred_list)    # 모델별 '주제의 정규화된 예측값'들의 평균 계산
    m_pred /= len(pred_list)   # 모델별 '분위기의 정규화된 예측값'들의 평균 계산
    e_pred /= len(pred_list)    # 모델별 '감정의 정규화된 예측값'들의 평균 계산
    s_pred /= len(pred_list)    # 모델별 '상황의 정규화된 예측값'들의 평균 계산 
    
    # 앙상블 결과 반환 
    pred = [t_pred, m_pred, e_pred, s_pred]    
    return pred


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


if __name__ == '__main__':
    main()