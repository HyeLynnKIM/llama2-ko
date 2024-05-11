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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=0
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
    # print(split_dot_train_data[:5])
    # print(split_dot_train_data[:10])
    all_splits = text_splitter.create_documents(split_dot_train_data)

    # for data in all_splits:
    #     if len(data.page_content) < 50:
    #         print(data)
    print('=================================================')
    # print(f'------Make split data Done!\nExample: "{all_splits[100]}" -------')
    # print(all_splits[10:15])
    # print(all_splits[1150:1160])
    # print(all_splits[5600:5610])
    # input()
    posts = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        # persist_directory=db_path[:11]
        persist_directory='./tmp_db',
        collection_metadata={'hnsw:space': 'cosine'}

    )
    # print(all_splits[9900])
    print(posts.similarity_search("충전이 더 잘되는 특허가 있나?"))
    input()

    return posts

def load_documents(db_path):
    emb_path = "BM-K/KoSimCSE-roberta"
    print('=================================================')
    print(f'------ Load DataBase -------')

    embeddings = HuggingFaceEmbeddings(
        model_name=emb_path, encode_kwargs={'normalize_embeddings': True}
    )

    # posts = Chroma(persist_directory=db_path[:11], embedding_function=embeddings)
    posts = Chroma(persist_directory='./tmp_db', embedding_function=embeddings)

    # print(posts.similarity_search_with_relevance_scores("배게", 10))
    # input()
    return posts


def run_chromadb(args):
    # db_path = './patent_db/chroma.sqlite3'
    db_path = './tmp_db/chroma.sqlite3'
    if os.path.isfile(db_path):
        print('=================================================')
        print(f'----------- DB is already Made! ---------------')
        posts = load_documents(db_path)
    else:
        print('=================================================')
        print(f'------------ DB is not already Made! ---------------')
        posts = preprocessing_documnets(db_path)

    print('=================================================')
    print('---------- Load Database Finish !! ----------------')

    task = 'text-generation'
    device_map = {"": 0}

    # prompt = PromptTemplate(
    #     template="""Your a smart patent assistant. You have to answer the question. If necessary, you can you given context. But, if you don't know about the question, PLEASE Says "I Don't Konw.". Answer must be KOREAN.
    #     ### User: {question}\n {context}
    #     ### Assistant:""",
    #     input_variables=["question", "context"]
    # )
    template = """Utilizing the context given below, answer the question.
    [context]
    {context}

    question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    # prompt = ChatPromptTemplate(
    #     messages=[
    #         SystemMessagePromptTemplate.from_template(
    #             "너는 유능한 어시스턴트야. 다음 질문에 대한 답변을 세 줄 이하로 작성해줘."
    #         ),
    #         # `variable_name`에 저장돼있는 건 'memory' 변수 안에 저장된것과 같아야함
    #         # MessagesPlaceholder(variable_name="chat_history"),
    #         HumanMessagePromptTemplate.from_template("{question}"),
    #     ]
    # )

    print('=================================================')
    print('--------------- Load Model ----- ---------')
    ## mdoel
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, load_in_4bit=True,
                                                 device_map=device_map, trust_remote_code=True)
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

    llm = HuggingFacePipeline(pipeline=pipe)
    # print(llm('질문에 대한 답변을 해줘. 모르면 모른다고 말해줘. ### 질문: 지구온난화란?'))
    # print('=======================')

    # memory는 대화 기록들을 저장할 수 있게 합니다. 유저가 새로운 질문을 했을때 전에 했던 대화 내용들을 바탕으로
    # 대답할 수 있게 하는거에요.

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    retriever = posts.as_retriever(
        serach_type='similarity_score_threshold',
        search_kwargs={"k": 2}
    )
    # tool = create_retriever_tool(
    #     retriever,
    #     "cusomter_service",
    #     "Searches and returns documents regarding the customer service guide.",
    # )
    #
    # tools = [tool]

    # ques = input('질문을 입력하세요.')
    ques = '카메라에 관한 특허 소개'
    # result = conversation({"question": ques})
    # print(result['text'])

    # print(f'------Check prompt:\n{prompt}')
    # print('=================================================')
    #
    #
    # print('----------- Make Pipeline ------------')
    #

    #
    # print(retriever.get_relevant_documents(ques))
    # print(posts.similarity_search(ques))
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # ?
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )
    # rag_chain = RunnableParallel(
    #         {"context": retreiver | format_docs, "query": RunnablePassthrough()})\
    #         | custom_rag_prompt \
    #         | hf \
    #         | StrOutputParser() \
    #
    # q = input('질문: ')
    # print(rag_chain.invoke(q))
    #
    # print('=================================================')

    result = qa_chain({'query': ques})

    print(result)

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