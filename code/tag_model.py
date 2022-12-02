import torch # 파이토치 패키지 임포트
import torch.nn as nn # 자주 사용하는 torch.nn패키지를 별칭 nn으로 명명
# 허깅페이스의 트랜스포머 패키지에서 BertConfig, BertModel 클래스 임포트
from transformers import BertConfig, BertModel

class TagClassifier(nn.Module):
    """
    노래 정보를 가지고 태그를 예측하는 모델
    """

    def __init__(self,cfg):
        """
        매개변수
        cfg: hidden_size, nlayers 등 설정값을 가지고 있는 변수 
        """

        super(TagClassifier,self).__init__()

        #설정값을 멤버 변수로 저장
        self.cfg=cfg
        
        #버트모델 설정값을 멤버변수로 저장
        self.bert_cfg=BertConfig(
            vocab_size=cfg.vocab_size,#사전 크기
            hidden_size=cfg.hidden_size,#히든 크기
            num_hidden_layers=cfg.nlayers,#레이어 층 수
            num_attention_heads=cfg.nheads,# 어텐션 헤드의 수
            intermediate_size=cfg.intermediate_size,#인텀미디어트 크기
            hidden_dropout_prob=cfg.dropout,#히든 드롭아웃 확률 값
            attention_probs_dropout_prob=cfg.dropout,#히든 드롭아웃 확률 값
            max_position_embeddings=cfg.seq_len,#포지션 임베딩의 최대 길이
            type_vocab_size=cfg.type_vocab_size,#타입 사전 크기
        )

        # 텍스트 인코더로 버트모델 사용
        self.text_encoder = BertModel(self.bert_cfg)

        # 음원 인코더로 CNN 사용
        self.sound_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
            
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            
            nn.Flatten(),
            nn.Linear(in_features=64*5*5,out_features=1024),
            nn.ReLU(),
            nn.Linear(1024,cfg.hidden_size)
        )

        # 분류기(Classifier) 생성기
        def get_clsfier(target_size=1):
            return nn.Sequential(
                nn.Linear(cfg.hidden_size*2,cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.Dropout(cfg.dropout),
                nn.ReLU(),
                nn.Linear(cfg.hidden_size,target_size)
            )
        
        # 노래 주제 분류기
        self.topic_clsfier=get_clsfier(cfg.n_t_cls)

        # 노래 분위기 분류기
        self.mood_clsfier=get_clsfier(cfg.n_m_cls)

        # 노래 감정 분류기
        self.emotion_clsfier=get_clsfier(cfg.n_e_cls)

        # 노래 상황 분류기(ex: 공부할때 듣는 노래,운동할때 듣는 노래,여행할때 듣는 노래)
        self.sit_clsfier=get_clsfier(cfg.n_s_cls)

    def forward(self,token_ids,token_mask,token_types,mel_spec,label=None):
        """
        매개변수
        token_ids : 전처리된 제목+가사 인덱스로 변환한 것
        token_mask : 실제 token_ids의 개수만큼 1, 나머지는 0으로 채움
        token_types : _문자를 기준으로 서로 다른 타입의 토큰임을 타입 인덱스로 저장
        mel_spec : 음원의 mel-spec
        label : 정답 주제/감정,분위기/상황 
        """

        # 전처리된 제목+가사를 하나의 텍스트 벡터(text_vec)로 변환
        # 반환 튜플(시퀀스 아웃풋,풀드(pooled) 아웃 풋) 중 시퀀스 아웃풋만 사용
        text_output = self.text_encoder(token_ids,token_mask,token_type_ids=token_types)[0]
        
        # 시퀀스 중 첫 타임 스탭의 hidden_state만 사용.
        text_vec=text_output[:,0]

        # mel-spec을 cnn에 넣어 sound_vec로 변환
        sound_vec=self.sound_encoder(mel_spec)

        # 음원벡터와 텍스트 벡터를 직렬연결(concatenate)하여 결합벡터 생성
        comb_vec=torch.cat([text_vec,sound_vec],1)

        # 결합된 벡터로 주제 예측
        t_pred = self.topic_clsfier(comb_vec)

        # 결합된 벡터로 분위기 예측
        m_pred = self.mood_clsfier(comb_vec)

        # 결합된 벡터로 감정 예측
        e_pred = self.env_clsfier(comb_vec)

        # 결합된 벡터로 상황 예측
        s_pred = self.sit_clsfier(comb_vec)

        # 데이터 패러럴 학습에서 GPU 메모리를 효율적으로 사용하기 위해 
        # loss를 모델 내에서 계산함.
        if label is not None:
            
            # 손실(loss) 함수로 CrossEntropyLoss를 사용
            # label의 값이 -1을 가지는 샘플은 loss계산에 사용 안 함
            loss_func = nn.CrossEntropyLoss(ignore_index=-1)
            
            # label은 batch_size x 3을 (batch_size x 1) 3개로 만듦
            t_label, m_label, e_label, s_label = label.split(1, 1)
            
            # 노래 주제를 예측한 값과 정답 값의 차이를 손실로 변환
            t_loss = loss_func(t_pred,t_label.view(-1))

            # 노래 분위기를 예측한 값과 정답 값의 차이를 손실로 변환
            m_loss = loss_func(m_pred,m_label.view(-1))

            # 노래의 감정을 예측한 값과 정답 값의 차이를 손실로 변환
            e_loss = loss_func(e_pred,e_label.view(-1))

            # 노래 듣는 상황을 예측한 값과 정답 값의 차이를 손실로 변환
            s_loss = loss_func(s_pred,s_label.view(-1))

            loss = t_loss + m_loss + e_loss + s_loss
        
        else: # label이 없으면 loss로 0을 반환
            loss = t_pred.new(1).fill_(0)
        
        # 최종 계산된 손실과 예측된 주제/감정,분위기/상황을 반환
        return loss,[t_pred,m_pred,e_pred,s_pred]

 

