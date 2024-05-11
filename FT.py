import os
import torch, random
import argparse
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import run_faiss, run_chroma

random.seed(42)
device_map = {"": 0}

def FT_main(args):
    ## 학습에 사용할 베이스 모델 경로
    model_name = args.model_name # llama-2-7b 모델을 샤딩한 버전입니다. 다른 llama-2 파일을 사용해도 무관합니다.

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
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.1,
        inference_mode=False,
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, eos_token='</s>')
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.unk_token

    # 학습 진행 중 loss가 치솟다가 0.0으로 떨어지는 문제 해결을 위해 사용
    tokenizer.padding_side = "right"

    # 구름 DS
    def format_kullm(sample):
        instruction = f"### 질문: {sample['instruction']}"
        input = f"{sample['input']}" if len(sample["input"]) > 0 else None
        output = f"### 답변\n{sample['output']}"
        # join all the parts together
        prompt = "\n\n".join([i for i in [instruction, input, output] if i is not None])
        return prompt

    # template dataset to add prompt to each sample
    def template_dataset(sample):
        sample["text"] = f"당신은 많은 지식을 가진 유능한 어시스턴트 입니다. 다음 질문에 대해서 올바르게 답해주세요.\n\n {format_kullm(sample)}{tokenizer.eos_token}"
        return sample

    # apply prompt template per sample
    dataset = load_dataset("nlpai-lab/kullm-v2", split="train")

    # Shuffle the dataset
    dataset_shuffled = dataset.shuffle(seed=42)
    dataset = dataset_shuffled.map(template_dataset, remove_columns=list(dataset.features))

    training_arguments = TrainingArguments(
        output_dir=args.save_path,
        save_steps=len(dataset)/args.epochs,
        save_total_limit=args.save_limit,  # 가장 최근 체크포인트 3개만 저장합니다.
        logging_steps=1000,
        learning_rate=args.lr,
        max_grad_norm=0.3,
        num_train_epochs=args.epochs,  # epochs 대신 max_steps을 기준으로 할 수 있습니다.
        # max_steps=args.max_steps,
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",  # paged_adamw_8bit 사용시 메모리를 더 절약할 수 있지만 loss가 0으로 떨어지는 문제가 있습니다.
        group_by_length=True,
        fp16=False,  # 코랩 무료버전에서 실행 시 "True"를 사용하세요
        bf16=True,  # 코랩 무료버전에서 실행 시 "False"를 사용하세요
        lr_scheduler_type="constant",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(args.save_path)

    model_to_save = trainer.model.module if hasattr(trainer.model,'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.save_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--model_name', type=str, help='모델 이름')
    parser.add_argument('--epochs', type=int, default=2, help='학습 에폭 수')
    parser.add_argument('--save_path', type=str, default='./model_save', help='저장 경로')
    parser.add_argument('--batch_size', type=int, default=2, help='배치 사이즈')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA Rank size')
    parser.add_argument('--save_limit', type=int, default=2, help='최근 기준 저장 체크포인트')
    parser.add_argument('--lr', type=int, default=1e-3, help='학습률')
    parser.add_argument('--max_steps', type=int, default=500, help='stpe수')

    args = parser.parse_args()
    FT_main(args)
    # "beomi/open-llama-2-ko-7b"