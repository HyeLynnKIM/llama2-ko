import os
import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
# from trl import SFTTrainer
# from guardrail.client import (
#     run_metrics,
#     run_simple_metrics,
#     create_dataset)

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings

import run_faiss, run_chroma

## api-key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '-'
repo_id = "beomi/open-llama-2-ko-7b"
task='text-generation'

local_rank = -1
device_map = {"": 0}

## 학습에 사용할 베이스 모델 경로
model_name = "beomi/open-llama-2-ko-7b" # llama-2-7b 모델을 샤딩한 버전입니다. 다른 llama-2 파일을 사용해도 무관합니다.

# 4bit QLoRA 학습을 위한 설정
bnb_4bit_compute_dtype = "bfloat16" # 코랩 무료버전에서 실행 시 "float16"를 사용하세요
bnb_4bit_quant_type = "nf4"
use_4bit = True
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
        print("=" * 80)

## 모델 불러오기
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    quantization_config=bnb_config
)

model.config.use_cache = False
model.config.pretraining_tp = 1

## 토크나이저 불러오기
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    inference_mode=False,
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 학습 진행 중 loss가 치솟다가 0.0으로 떨어지는 문제 해결을 위해 사용
tokenizer.padding_side = "right"

## 테스트
def gen(x):
    gened = model.generate(
        **tokenizer(
            x,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=256,
        # no_repeat_ngram_size=6,
        # top_p=0.8,
        # temperature=0.7,
        # early_stopping=True,
        # num_return_sequences=5,
        do_sample=True,
        eos_token_id=2,
        pad_token_id=2
    )
    # print(gened[0])
    print(tokenizer.decode(gened[0]))

# Inference and evaluate outputs/prompts
# prompt = "다음 질문에 대해 설명하기: 감기"
# gen(prompt)

## data
data = []
data.append('네트워크 형 오티피 제공 방법(Method for Providing Network type OTP)')
data.append('본 발명은 서버에 의해 실행되는 방법에 있어서, 특정 프로그램을 구비한 사용자 휴대폰을 지정된 인증 절차의 OTP(One Time Password)를 구현하는 사용자의 OTP 매체임을 인증하기 위해 중복되지 않는 코드값 형태로 상기 사용자 휴대폰의 지정된 저장영역에 저장된 별도의 매체 식별자를 포함하는 매체 정보를 확인하여 지정된 저장매체에 저장시키는 절차를 수행하고, 상기 사용자 휴대폰에서 상기 특정 프로그램을 구동 시, 상기 사용자 휴대폰에서 구동된 특정 프로그램과 통신하여 상기 특정 프로그램을 통해 상기 사용자 휴대폰의 지정된 저장영역으로부터 추출된 별도의 매체 식별자를 포함하는 매체인증 정보를 수신하고 상기 저장된 매체 정보를 통해 상기 매체인증 정보에 포함된 상기 별도의 매체 식별자를 비교 인증하여 상기 특정 프로그램을 구동한 사용자 휴대폰을 사용자의 OTP 매체로 인증하는 절차를 수행하고, 상기 특정 프로그램을 구동한 사용자 휴대폰을 사용자의 OTP 매체로 인증 시, 상기 인증된 사용자의 OTP 매체로 제공하기 위해 서버 측에서 동적으로 생성된 N(N≥2)자리수의 OTP를 확인하고, 상기 인증된 사용자의 OTP 매체에 대응하는 사용자 휴대폰의 특정 프로그램으로 상기 N자릿수의 OTP를 제공하는 절차를 수행하며, 상기 별도의 매체 식별자는, 상기 휴대폰의 기 설정된 시스템 영역에 저장된 전화번호와 일련번호가 아닌 별도의 중복되지 않는 코드값인 네트워크 형 오티피 제공 방법에 관한 것이다.')
data.append('그리고, 본 발명의 일 실시예에 따른 영상 복원 시스템은 500MHz에서 3GHz까지 UHF 대역 전자파뿐만 아니라, 다른 주파수 대역을 갖는 전자파를 이용할 수 있다.')
pd.DataFrame(data).to_csv("train_data.csv", index=False, encoding='utf-8')

## embedding
modelPath = "all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


run_chroma.run_chromadb(model=model, tokenizer=tokenizer, file_path='train_data.csv', task=task, embeddings=embeddings)
