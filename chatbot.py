from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import streamlit as st
import time
import openai
import jsonlines
import re, os


from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool

os.environ['OPENAI_API_KEY'] = '-'


def preprocessing_json_file(dir: str):
    data = []
    i = 0
    with jsonlines.open(dir) as file:
        for line in file.iter():
            if i == 20000: break
            i += 1
            inp = f"제목: {line['input'].split(': ')[1]}"
            out = re.sub(r'[(0-9)]', '',line['output'])
            out = re.sub(r'  ', '', out)
            data.append(f"{inp} \n\n {out}")

    # txt저장
    s_path = './data/txt_patent2.txt'
    with open(s_path, 'w', encoding='utf-8') as file2:
        file2.write('\n'.join(data))

# LLM 셋업하기
llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                 max_token=512,
                 temperature=0)

preprocessing_json_file('./data/total_patent_data.jsonl')
# WebBaseLoader 함수가 " " 안에 있는 웹사이트 텍스트들을 불러옵니다.
loader = TextLoader(file_path='./data/txt_patent2.txt', encoding='utf-8')
data = loader.load()

# 불러온 데이터 스플릿하기
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
all_splits = text_splitter.split_documents(data)

# 불러온 데이터 Chroma라는 vector store에 저장하기.
# LLM 에이전트가 이 데이터베이스에 저장돼 있는 내용을 검색할 수 있도록 해주는 겁니다.
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings()
)

# LLM 에이전트가 사용할 수 있도록 retriever 라는 툴로 만들어줍니다.
retriever = vectorstore.as_retriever()

tool = create_retriever_tool(
    retriever,
"search_patent",
"Searches and returns documents regarding the patent information.",
)

tools = [tool]

memory_key = "history"

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)

memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

# LLM이 어떤 행동을 해야하는지 알려주는 프롬프트
system_message = SystemMessage(
    content=(
        # 여기서는 챗봇에게 너는 어떤 챗봇인지 설명해주고, 위에서 만든 툴을 활용해서 답변을 하라고 해줍니다
        "You are a nice patent information agent."
        "Do your best to answer the questions."
        "Feel free to use any tools available to look up "
        "relevant information, only if necessary"
        # 할루시네이션 전처리 과정입니다. 프롬프트로 "만약 서비스 매뉴얼과 관련되지 않은 질문이면 거짓된 답변을 생성하지 말아줘" 라고 처리합니다.
        "Do not generate false answers to questions that are not related to the customer service guide."
        "If you don't know the answer, just say that you don't know. Don't try to make up an answer."
        # 챗봇이 한국어로 대답하게 합니다.
        "Make sure to answer in Korean."
        "Answer line length should be in four."
    )
)

# 위에서 만든 프롬프트를 바탕으로 LLM에게 줄 프롬프트 형식 셋업
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

# 위에서 만든 llm, tools, prompt를 바탕으로 에이전트 만들어주기
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

result = agent_executor({"input": "안녕하세요"})
print(result)