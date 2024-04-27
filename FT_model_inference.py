import argparse
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
from peft import PeftModel

def gen(x, model, tokenizer):
    prompt = f"""당신은 많은 지식을 가진 유능한 어시스턴트 입니다. 주어지는 질문에 대한 답변을 해주세요. 단, 답변은 세 줄까지만 출력하세요.
    ### 질문: {x}\n\n### 답변:
    """
    gened = model.generate(
        **tokenizer(
            prompt,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=256,
        no_repeat_ngram_size=6,
        # top_p=0.8,
        temperature=0.3,
        early_stopping=True,
        num_return_sequences=5,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.pad_token_id
    )
    # print(gened[0])
    print(tokenizer.decode(gened[0]))

def inference(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, load_in_8bit=False, device_map="auto", trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(model, args.save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(tokenizer.eos_token)
    ques = input('질문을 입력하세요.')
    gen(ques, peft_model, tokenizer)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--model_name', type=str, help='모델 이름')
    parser.add_argument('--save_path', type=str, default='./model_save/checkpoint-76315', help='저장 경로')

    args = parser.parse_args()
    inference(args)
