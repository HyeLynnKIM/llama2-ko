import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate

import argparse
import json, jsonlines
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,

)
from langchain.schema.messages import SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_toolkits import create_retriever_tool
# from langchain.agents.
from langchain.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate

def preprocessing_json_file(dir: str):
    data = []
    with jsonlines.open(dir) as file:
        for line in file.iter():
            inp = f"제목: {line['input'].split(': ')[1]}"
            out = re.sub(r'[(0-9)]', '',line['output'])
            out = re.sub(r'  ', '', out)
            data.append(f"{inp} \n\n {out}")

    # txt저장
    s_path = './data/txt_patent.txt'
    with open(s_path, 'w', encoding='utf-8') as file2:
        file2.write('\n\n'.join(data))

def preprocessing_documnets(db_path):
    emb_path = "BM-K/KoSimCSE-roberta"
    print('=================================================')
    print(f'------Make Embeddings: "{emb_path}" --------')

    embeddings = HuggingFaceEmbeddings(
        model_name=emb_path, encode_kwargs={'normalize_embeddings': True}
    )

    loader = TextLoader(file_path='./data/txt_patent.txt', encoding='utf-8')
    train_data = loader.load()
    train_data = train_data[0].page_content.split('\n\n')
    split_dot_train_data = []
    # for text in train_data:
    #     split_dot_train_data += text.split('. ')
    # split_dot_train_data_ = []
    # print(train_data[:10])
    i=0
    while i < len(train_data):
        tmp_text = ''
        if '제목: ' in train_data[i]:
            tmp_text += train_data[i]
            tmp_text += '- '
            while i < len(train_data):
                i += 1
                if i >= len(train_data) or '제목: ' in train_data[i]: break
                tmp_text += train_data[i]
            split_dot_train_data.append(tmp_text + '.')
    print(split_dot_train_data[-1])
    print('=================================================')
    print(f'------ Make split data Done! -------')

    db = FAISS.from_texts(split_dot_train_data, embedding=embeddings)
    db.save_local("faiss_index")

    return db

def load_documents(db_path):
    emb_path = "BM-K/KoSimCSE-roberta"
    print('=================================================')
    print(f'------ Load DataBase -------')

    embeddings = HuggingFaceEmbeddings(
        model_name=emb_path, encode_kwargs={'normalize_embeddings': True}
    )
    db = FAISS.load_local(db_path[:13], embeddings=embeddings, allow_dangerous_deserialization=True)

    return db

def run_chromadb(args):
    db_path = './faiss_index/index.faiss'
    if os.path.isfile(db_path):
        print('=================================================')
        print(f'----------- DB is already Made! ---------------')
        db = load_documents(db_path)
    else:
        print('=================================================')
        print(f'--------- DB is not already Made! ------------')
        db = preprocessing_documnets(db_path)

    print('=================================================')
    print('---------- Load Database Finish !! ----------------')

    task = 'text-generation'
    device_map = {"": 0}

    print('=================================================')
    print('--------- Load Model --------------------')

    ## mdoel
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map=device_map,
        trust_remote_code=True
    )
    peft_model = PeftModel.from_pretrained(model, args.save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    pipe = pipeline(
        task=task,
        model=peft_model,
        tokenizer=tokenizer,
        max_length=1024,
        # eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.unk_token_id,
        do_sample=True,
        # num_return_sequences=,
        # top_k=5,
        temperature=0.2,
        no_repeat_ngram_size=3,
        # early_stopping=True
    )
    retreiver = db.as_retriever(
        search_kwargs={"k": 2},
        # search_type="mmr"
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Utilizing the context given below, answer the question in Korean.
    [context]
    {context}
    
    question: {query}
    """

    custom_rag_prompt = ChatPromptTemplate.from_template(template)

    rag_chain = RunnableParallel(
            {"context": retreiver | format_docs, "query": RunnablePassthrough()})\
            | custom_rag_prompt \
            | hf \
            | StrOutputParser() \

    q = input('질문: ')
    print(rag_chain.invoke(q))
    print(hf(q))
    # for chunk in rag_chain.stream(q):
    #     print(chunk, end='', flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--model_name', type=str, help='모델 이름')
    parser.add_argument('--save_path', type=str, default='./model_save/checkpoint-76315', help='저장 경로')
    parser.add_argument('--file_path', type=str, default='./data', help='파일 경로')
    parser.add_argument('--db_path', type=str, help='파일 경로')
    args = parser.parse_args()

    data_path = './data/txt_patent.txt'
    if os.path.isfile(data_path):
        print('=================================================')
        print(f'------ "{data_path}" is available ------ Keep Inference!')
    else:
        print('=================================================')
        print(f'------ "{data_path}" is NOT available ------ Make data!')
        preprocessing_json_file(args.file_path)
        print(f'------ Made data ,,,\n{data_path} is available ------ Keep Inference!')

    run_chromadb(args)