from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import random
import logging
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import json
from transformers import DefaultDataCollator
import pandas as pd

df = pd.DataFrame(columns=['question', 'answer', 'answer_start', 'answer_end', ])

random.seed(90)

with open(r"C:\Users\helen\Downloads\KorQuAD_v1.0_train.json", "r", encoding='utf-8') as read_file:
    train = json.load(read_file)
with open(r"C:\Users\helen\Downloads\KorQuAD_v1.0_dev.json", "r", encoding='utf-8') as read_file:
    test = json.load(read_file)

start_positions = []
end_positions = []
answers = []

for item in train['data']:
    for para in item['paragraphs']:
        print(para['context'])
        for qas in para['qas']:
            answer = qas['answers'][0]['']
    exit(0)

train = train['data']
print(train[0])
model = BertForQuestionAnswering.from_pretrained('klue/bert-base')
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

## ft
#train_args are the parameters the QuestionAnswerringModel will use
data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="./qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

question = "면역 체계는 무엇입니까?"
paragraph = "면역 체계는 질병으로부터 보호하는 유기체 내의 다양한 생물학적 구조와 과정의 시스템입니다. 제대로 기능하려면 면역 체계가 바이러스에서 기생충에 이르기까지 병원균으로 알려진 다양한 물질을 탐지하고 유기체의 건강한 조직과 구별해야 합니다."

# 인코딩
encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)

inputs = encoding['input_ids']  #Token embeddings
sentence_embedding = encoding['token_type_ids']  #Segment embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

# segment_ids = [0] * len(question_tokens)
# segment_ids += [1] * len(paragraph_tokens)
# print(len(input_ids), len(segment_ids)) # 같아야 한다.

outputs = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))

start_index = torch.argmax(outputs['start_logits'])
end_index = torch.argmax(outputs['end_logits'])
answer = ' '.join(tokens[start_index:end_index+1])

print(tokens)
print(start_index, end_index)
print('Q :', question)
print('A :', answer.replace('##', ''))