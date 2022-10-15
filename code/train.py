import os
import time
import math
import torch
import random
import argparse
import tag_dataset 
import tag_model
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import AdamW, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings(action='ignore')

# 전처리 전 데이터가 저장된 디렉토리
RAW_DATA_DIR="../input/raw_data"

# 전처리된 데이터가 저장된 디렉터리
DB_PATH=f'../input/processed'

# 토큰을 인덱스로 치환할 때 사용될 사전 파일이 저장된 디렉터리 
VOCAB_DIR=os.path.join(DB_PATH, 'vocab')

# 학습된 모델의 파라미터가 저장될 디렉터리
MODEL_PATH=f'../model'

# 미리 정의된 설정 값
class CFG:
    learning_rate=3.0e-4 # 러닝 레이트
    batch_size=256 # 배치 사이즈
    num_workers=4 # 워커의 개수
    print_freq=100 # 결과 출력 빈도
    start_epoch=0 # 시작 에폭
    num_train_epochs=10 # 학습할 에폭수
    warmup_steps=100 # lr을 서서히 증가시킬 step 수
    max_grad_norm=10 # 그래디언트 클리핑에 사용
    weight_decay=0.01
    dropout=0.2 # dropout 확률
    hidden_size=512 # 은닉 크기
    intermediate_size=256 # TRANSFORMER셀의 intermediate 크기
    nlayers=2 # BERT의 층수
    nheads=8 # BERT의 head 개수
    seq_len=256 # 토큰의 최대 길이
    n_t_cls = 5 # 노래주제 개수 - 아직 정확히 안 정해짐
    n_m_cls = 6 # 감정,분위기 개수 - 아직 정확히 안 정해짐
    n_s_cls = 3 # 상황 개수 - 아직 정확히 안 정해짐
    vocab_size = 32000 # 토큰의 유니크 인덱스 개수
    type_vocab_size = 1000 # 타입의 유니크 인덱스 개수
    train_path = os.path.join(DB_PATH,'train.json')
    mel_spec_path = os.path.join(DB_PATH,'music/mel')

def main():
    # 명령행에서 받을 키워드 인자를 설정한다.
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='')        
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)    
    parser.add_argument("--nepochs", type=int, default=CFG.num_train_epochs)    
    parser.add_argument("--seq_len", type=int, default=CFG.seq_len)
    parser.add_argument("--nworkers", type=int, default=CFG.num_workers)
    parser.add_argument("--wsteps", type=int, default=CFG.warmup_steps)
    parser.add_argument("--seed", type=int, default=7)        
    parser.add_argument("--nlayers", type=int, default=CFG.nlayers)
    parser.add_argument("--nheads", type=int, default=CFG.nheads)
    parser.add_argument("--hidden_size", type=int, default=CFG.hidden_size)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--stratified", action='store_true')
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)    
    args = parser.parse_args()

    # 키워드 인자로 받은 값을 CFG로 다시 저장합니다.
    CFG.batch_size=args.batch_size
    CFG.num_train_epochs=args.nepochs
    CFG.seq_len=args.seq_len
    CFG.num_workers=args.nworkers
    CFG.warmup_steps=args.wsteps    
    CFG.learning_rate=args.lr
    CFG.dropout=args.dropout
    CFG.seed =  args.seed        
    CFG.nlayers =  args.nlayers    
    CFG.nheads =  args.nheads
    CFG.hidden_size =  args.hidden_size    
    print(CFG.__dict__)
    
    # 전처리되기 전 데이터 읽어와 분류해야할 수를 가져온다.
    raw_df=pd.read_json(os.path.join(RAW_DATA_DIR,'train.json'))
    CFG.n_t_cls=len(raw_df['topic'].astype('category').cat.categories)
    CFG.n_m_cls=len(raw_df['mood'].astype('category').cat.categories)
    CFG.n_s_cls=len(raw_df['situation'].astype('category').cat.categories)
    

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)    
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True

    # 전처리된 데이터를 읽어옴
    print('loading ...')
    train_df = pd.read_json(CFG.train_path)

    # StratifiedKFold 사용
    if args.stratified:
        print('use StratifiedKFold ...')
        
        # StratifiedKFold를 사용해 데이터셋을 학습셋(train_df)과 검증셋(valid_df)으로 나눔
        folds = StratifiedKFold(n_splits=5,random_state=CFG.seed,shuffle=True)
        train_idx,valid_idx = list(folds.split(train_df.values,train_df['unique_tagid']))[args.fold]
    
    else:
        # KFold를 사용해 데이터셋을 학습셋(train_df)과 검증셋(valid_df)으로 나눔
        folds = KFold(n_splits=5,random_state=CFG.seed,shuffle=True)
        train_idx,valid_idx = list(folds.split(train_df.values))[args.fold]
    
    valid_df = train_df.iloc[valid_idx]
    train_df = train_df.iloc[train_idx]

    # 토큰을 대응되는 인덱스로 치환할 때 사용될 딕셔너리를 로딩
    vocab = [line.split('\t')[0] for line in open(os.path.join(VOCAB_DIR,
                                'spm.vocab'),encoding='utf-8').readlines()]
    
    token2id = dict([(w,i) for i,w in enumerate(vocab)])
    print('loading ... done')

    # 학습에 적합한 형태의 샘플을 가져오는 TagDataset의 인스턴스를 만듦
    train_db = tag_dataset.TagDataset(train_df,CFG.mel_spec_path,token2id,
                                        CFG.seq_len,CFG.type_vocab_size)
    valid_db = tag_dataset.TagDataset(valid_df,CFG.mel_spec_path,token2id,
                                        CFG.seq_len,CFG.type_vocab_size)

    # 여러 개의 워커로 빠르게 배치(미니배치)를 생성하도록 함
    train_loader = DataLoader(
        train_db,batch_size=CFG.batch_size,shuffle=True,drop_last=True,
        num_workers=CFG.num_workers,pin_memory=False)
    
    valid_loader = DataLoader(
        valid_db,batch_size=CFG.batch_size,shuffle=True,drop_last=True,
        num_workers=CFG.num_workers,pin_memory=False)

    # 카테고리 분류기 모델을 생성함
    model = tag_model.TagClassifier(CFG)

    # 쿠다 사용가능한지 확인
    if not torch.cuda.is_available():
        print('Gpu 사용할 수 없습니다.')
        exit()
    
    # 모델의 파라미터를 GPU 메모리로 옮김
    model.cuda()

    # 모델의 파라미터 수를 출력
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('parameters: ',count_parameters(model))

    # GPU가 2개 이상이면 데이터 패럴렐로 학습 가능하게 만듦
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # 학습 동안 수행될 총 step 수
    # 데이터 셋을 배치 크기로 나눈 것이 1 epoch 동안의 step 수
    # 총 스텝 수 = 1 epoch step 수 * 총 epoch
    num_train_optimization_steps = int(
        len(train_db) / CFG.batch_size * (CFG.num_train_epochs))
    print('num_train_optimization_steps',num_train_optimization_steps)

    # 파라미터 그루핑 정보 생성
    # 가중치 감쇠(weight decay) 미적용 파라미터 그룹과 적용 파라미터로 나눔
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd
            in no_decay)],'weight_decay':0.01},
        {'params':[p for n,p in param_optimizer if any(nd in n for nd
            in no_decay)],'weight_decay':0.0}
    ]

    # AdamW 옵티마이저 생성
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=CFG.learning_rate,
                        weight_decay=CFG.weight_decay,
                        )
    
    # learning_rate가 선형적으로 감소하는 스케줄러 생성
    scheduler = get_linear_schedule_with_warmup(optimizer,
                            num_warmup_steps=CFG.warmup_steps,
                            num_training_steps=num_train_optimization_steps)
    print('use WarmupLinearSchedule ...')

    def get_lr():
        return scheduler.get_lr()[0]
    
    # epoch 별 실험결과 로그를 저장할 데이터 프레임
    log_df = pd.DataFrame() 
    curr_lr = get_lr()
    print(f'Initial learning rate : {curr_lr}')

    # (num_train_epochs - start _epoch) 횟수만큼 학습을 진행
    for epoch in range(CFG.start_epoch,CFG.num_train_epochs):

        # 한 epoch의 결과가 집계된 한 행을 반환
        def get_log_row_df(epoch,lr,train_res,valid_res):
            log_row={
                'EPOCH':epoch,'LR':lr,
                'TRAIN_LOSS':train_res[0], 'TRAIN_TACC':train_res[1],
                'TRAIN_MACC':train_res[2], 'TRAIN_SACC':train_res[3],
                
                'VALID_LOSS':valid_res[0], 'VALID_TACC':valid_res[1],
                'VALID_MACC':valid_res[2], 'VALID_SACC':valid_res[3],
            }
            return pd.DataFrame(log_row,index=[0])
        
        # 학습을 진행하고 loss나 acc와 같은 결과를 반환합니다.
        train_res = train(train_loader,model,optimizer,epoch,scheduler)

        # 검증을 진행하고 loss나 acc와 같은 결과를 반환합니다.
        valid_res = validate(valid_loader,model)

        curr_lr = get_lr()
        print(f'set the learning_rate : {curr_lr}')

        log_row_df = get_log_row_df(epoch,curr_lr,train_res,valid_res)

        # log_df에 결과가 집계된 한 행을 추가합니다.
        log_df = log_df.append(log_row_df,sort=False)
        print(log_df.tail(10)) # log_df 의 최신 10개 행만 출력합니다.

        # 모델의 파라미터가 저장될 파일의 이름을 정합니다.
        curr_model_name = (f'b{CFG.batch_size}_h{CFG.hidden_size}_'
                            f'd{CFG.dropout}_l{CFG.nlayers}_hd{CFG.nheads}_'
                            f'ep{epoch}_s{CFG.seed}_fold{args.fold}.pt')
        
        # torch.nn.DataParallel로 감싸진 경우 원래의 model를 가져옵니다.
        model_to_save = model.module if hasattr(model, 'module') else model

    print('training done')

    # 모델의 파라미터를 저장합니다.
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': 'transformer',
        'state_dict': model_to_save.state_dict(),
        'log': log_df,
        },
        MODEL_PATH, curr_model_name,
    )  

def train(train_loader,model,optimizer,epoch,scheduler):
    """
    한 epoch 학습합니다.
    """
    batch_time = AverageMeter() # 배치처리 시간집계
    data_time = AverageMeter()      # 데이터 로딩 시간 집계
    losses = AverageMeter()         # 손실 값 집계
    t_accuracies = AverageMeter()   # 주제 정확도 집계
    m_accuracies = AverageMeter()   # 감정,분위기 정확도 집계
    s_accuracies = AverageMeter()   # 상황 정확도 집계

    sent_count = AverageMeter()     # 문장 처리 개수 집계

    # 학습 모드로 교체
    model.train()

    start = end = time.time()
    # train_loader에서 반복해서 학습용 배치 데이터를 받아옵니다.
    # TagDataset의 __getitem__() 함수의 반환 값과 동일한 변수 반환
    for step, (token_ids, token_mask, token_types, img_feat, label) in enumerate(train_loader):
        # 데이터 로딩 시간 기록
        data_time.update(time.time() - end)
        
        # 배치 데이터의 위치를 CPU메모리에서 GPU메모리로 이동
        token_ids, token_mask, token_types, img_feat, label = (
            token_ids.cuda(), token_mask.cuda(), token_types.cuda(), 
            img_feat.cuda(), label.cuda())
                
        batch_size = token_ids.size(0)   
                
        # model은 배치 데이터를 입력 받아서 예측 결과 및 loss 반환
        # model은 인스턴스이나 __call__함수가 추가돼 함수처럼 호출이 가능합니다. 
        # TagClassifier의 __call__ 함수 내에서 forward 함수가 호출됩니다. 
        loss, pred = model(token_ids, token_mask, token_types, img_feat, label)
        loss = loss.mean() # Multi-GPU 학습의 경우 mean() 호출 필요
                
        # loss 값을 기록
        losses.update(loss.item(), batch_size)
        
        # 역전파 수행
        loss.backward()

        # CFG.max_grad_norm 이상의 값을 가지는 그래디언트 값 클리핑
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
              
        scheduler.step()    # 스케쥴러로 learning_rate 조절
        optimizer.step()    # 옵티마이저로 파라미터 업데이터
        optimizer.zero_grad() # 옵티마이저 내의 그래디언트 초기화

        # 소요시간 측정
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        # CFG.print_freq 주기대로 결과 로그를 출력
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            # 주제/감정,분위기/상황이 예측된 pred와 정답 label로 정확도 계산 및 집계
            t_acc, m_acc, s_acc = calc_tag_acc(pred, label)
            t_accuracies.update(t_acc, batch_size)
            m_accuracies.update(m_acc, batch_size)
            s_accuracies.update(s_acc, batch_size)
        
            
            print(
                'Epoch: [{0}][{1}/{2}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Elapsed {remain:s} '
                'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                'TAcc: {t_acc.val:.3f}({t_acc.avg:.3f}) '
                'MAcc: {m_acc.val:.4f}({m_acc.avg:.3f}) '
                'SAcc: {s_acc.val:.3f}({s_acc.avg:.3f}) '                  
                'Grad: {grad_norm:.4f}  '
                'LR: {lr:.6f}  '
                'sent/s {sent_s:.0f} '
                .format(
                epoch, step+1, len(train_loader),
                data_time=data_time, loss=losses,
                t_acc=t_accuracies, m_acc=m_accuracies,
                s_acc=s_accuracies, 
                remain=timeSince(start, float(step+1)/len(train_loader)),
                grad_norm=grad_norm,
                lr=scheduler.get_lr()[0],                   
                sent_s=sent_count.avg/batch_time.avg
                )
            )
    # 학습 동안 집계된 결과 반환
    return (losses.avg, t_accuracies.avg, m_accuracies.avg, 
            s_accuracies.avg)

def validate(valid_loader,model):
    """    
    한 에폭 단위로 검증합니다.
    매개변수
    valid_loader: 검증 데이터셋에서 배치(미니배치) 불러옵니다.
    model: train 함수에서 학습된 딥러닝 모델
    """    
    batch_time = AverageMeter()     # 한 배치처리 시간 집계
    data_time = AverageMeter()      # 데이터 로딩 시간 집계
    losses = AverageMeter()         # 손실 값 집계
    t_accuracies = AverageMeter()   # 주제 정확도 집계
    m_accuracies = AverageMeter()   # 감정,분위기 정확도 집계
    s_accuracies = AverageMeter()   # 상황 정확도 집계
    
    sent_count = AverageMeter()     # 문장 처리 개수 집계
    
    # 평가(evaluation) 모드로 교체
    # 드롭아웃이나 배치정규화가 일관된 값을 내도록 함
    model.eval()

    start = end = time.time()
        
    for step, (token_ids, token_mask, token_types, img_feat, label) in enumerate(valid_loader):
        # 데이터 로딩 시간 기록
        data_time.update(time.time() - end)
        
        # 배치 데이터의 위치를 CPU메모리에서 GPU메모리로 이동
        token_ids, token_mask, token_types, img_feat, label = (
            token_ids.cuda(), token_mask.cuda(), token_types.cuda(), 
            img_feat.cuda(), label.cuda())
        
        batch_size = token_ids.size(0)
        
        # with문 내에서는 그래디언트 계산을 하지 않도록 함
        with torch.no_grad():
            # model은 배치 데이터를 입력 받아서 예측 결과 및 loss 반환
            loss, pred = model(token_ids, token_mask, token_types, img_feat, label)
            loss = loss.mean()
                
        # loss 값을 기록
        losses.update(loss.item(), batch_size)
        
        # 소요시간 측정
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        # CFG.print_freq 주기대로 결과 로그를 출력
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            t_acc, m_acc, s_acc = calc_tag_acc(pred, label)

            t_accuracies.update(t_acc, batch_size)
            m_accuracies.update(m_acc, batch_size)
            s_accuracies.update(s_acc, batch_size)
            
            print(
                'TEST: [{0}/{1}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Elapsed {remain:s} '
                'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                'TAcc: {t_acc.val:.3f}({t_acc.avg:.3f}) '
                'MAcc: {m_acc.val:.4f}({m_acc.avg:.3f}) '
                'SAcc: {s_acc.val:.3f}({s_acc.avg:.3f}) '
                'sent/s {sent_s:.0f} '
                .format(
                step+1, len(valid_loader),
                data_time=data_time, loss=losses,
                t_acc=t_accuracies, m_acc=m_accuracies,
                s_acc=s_accuracies,
                remain=timeSince(start, float(step+1)/len(valid_loader)),
                sent_s=sent_count.avg/batch_time.avg
                ))
    # 검증 동안 집계된 결과 반환
    return (losses.avg, t_accuracies.avg, m_accuracies.avg, 
            s_accuracies.avg)

def calc_tag_acc(pred, label):
    """
    주제/감정,분위기/상황 태그별 정확도와 전체(overall) 정확도를 반환
    """
    t_pred, m_pred, s_pred = pred    
    _, t_idx = t_pred.max(1)
    _, m_idx = m_pred.max(1)
    _, s_idx = s_pred.max(1)
        
    t_acc = (t_idx == label[:, 0]).sum().item() / (label[:, 0]>=0).sum().item()
    m_acc = (m_idx == label[:, 1]).sum().item() / (label[:, 1]>=0).sum().item()        
    s_acc = (s_idx == label[:, 2]).sum().item() / (label[:, 2]>=0).sum().item()
        
    return t_acc, m_acc, s_acc

def save_checkpoint(state,model_path,model_filename,is_best=False):
    print('saving current state model ...')
    
    # 모델 저장 폴더가 없다면 폴더를 생성한다.
    if not os.path.exists(model_path):
        os.makedirs(model_path)
     
    if is_best:
        # 지금까지의 최고 성능의 모델인 경우 앞에 best_를 붙여 저장한다.
        torch.sae(state,os.path.join(model_path,'best_'+model_filename))
    else:
        # 경로에 모델을 저장한다.
        torch.save(state,os.path.join(model_path,model_filename))   

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