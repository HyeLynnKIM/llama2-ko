from langchain.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate

def run_faiss(model, tokenizer, file_path, embeddings, device_map):
    '''
    :param model: model, especially huggingface model
    :param tokenizer: tokenizer
    :param file_path: file path, like csv or txt
    :param embeddings:
    :param device_map:
    :return:
    '''

    loader = CSVLoader(file_path=file_path, encoding='utf-8')
    train_data = loader.load()

    db = FAISS.from_documents(train_data, embedding=embeddings)
    db.save_local("faiss_index")

    # db = FAISS.load_local("faiss_index", embeddings)
    retreiver = db.as_retriever(search_kwargs={"k": 4})

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device=0)
    hf = HuggingFacePipeline(pipeline=pipe)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """마지막에 질문에 답하려면 다음과 같은 맥락을 사용합니다.

    {context}"

    질문: {question}

    유용한 답변:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
            {"context": retreiver | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | hf
            | StrOutputParser()
    )

    for chunk in rag_chain.stream('네트워크 형 오티피에 관한 특허 설명해줘'):
        print(chunk, end='', flush=True)