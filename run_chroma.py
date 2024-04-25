import chromadb
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate

def run_chromadb(model, tokenizer, file_path, embeddings, task, device_map):
    '''
    :param model:
    :param tokenizer:
    :param file_path:
    :param embeddings:
    :param device_map:
    :return:
    '''
    # client = chromadb.PersistentClient()
    # embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    #     model_name=modelPath
    # )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0
    )

    loader = CSVLoader(file_path=file_path, encoding='utf-8')
    train_data = loader.load()

    all_splits = text_splitter.split_documents(train_data)
    print(all_splits[0])

    posts = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings
    )

    prompt = PromptTemplate(
        template="""너는 다음 질문에 대한 답을 본문을 기반으로 해야 해.\n질문: {question}
        {context}
        답: """,
        input_variables=["question", "context"]
    )
    # print(prompt)

    ##
    # llm = LlamaCpp(
    #     # model_path: 로컬머신에 다운로드 받은 모델의 위치
    #     model_path=model_name,
    #     temperature=0.75,
    #     top_p=0.95,
    #     max_tokens=8192,
    #     verbose=True,
    #     # n_ctx: 모델이 한 번에 처리할 수 있는 최대 컨텍스트 길이
    #     n_ctx=8192,
    #     # n_gpu_layers: 실리콘 맥에서는 1이면 충분하다고 한다
    #     n_gpu_layers=1,
    #     n_batch=512,
    #     f16_kv=True,
    #     n_threads=16,
    # )

    # llm = HuggingFaceHub(
    #     repo_id=repo_id,
    #     task=task,
    #     model_kwargs={
    #         "temperature": 0.75,
    #         "max_tokens": 512,
    #         "verbose": True,
    #     }
    # )

    pipe = pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map=device_map,
        max_length=512,
        do_sample=True,
        top_k=7,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=posts.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    question = 'OTP는 무슨 글자의 약자야?'
    result = qa_chain({'query': question, 'context': qa_chain.retriever})

    # result = posts.query(
    #     query_texts=[],
    #     n_results=4
    # )

    print(result)