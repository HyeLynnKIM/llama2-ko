import chromadb
import time
import random
import torch

import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter

from chromadb.utils import embedding_functions
import json

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import HTML_Processor
from langchain_community.document_transformers import LongContextReorder

random.seed(99)

import openai
from rag_utils import *


## 검색 모델
def no_pre_model():
    # 필요 변수들
    cnt = 0
    score1, score2, score3, score4 = 0, 0, 0, 0# 총점수 저장
    ques_cnt = 0 #질문 총 개수

    # topk 시 3가지 저장 변수
    cor_top1, not_cor_top1 = 0, 0
    cor_top2, not_cor_top2 = 0, 0
    cor_top3, not_cor_top3 = 0, 0

    # reorder - topk
    cor_re_top1, not_cor_re_top1 = 0, 0
    cor_re_top2, not_cor_re_top2 = 0, 0
    cor_re_top3, not_cor_re_top3 = 0, 0

    # reorder - topk - rerank
    cor_re_top1_re, not_cor_re_top1_re = 0, 0
    cor_re_top2_re, not_cor_re_top2_re = 0, 0
    cor_re_top3_re, not_cor_re_top3_re = 0, 0

    # topk - rerank
    cor_top1_re, not_cor_top1_re = 0, 0
    cor_top2_re, not_cor_top2_re = 0, 0
    cor_top3_re, not_cor_top3_re = 0, 0

    times1, times2, times3, times4 = 0.0 ,0.0, 0.0, 0.0
    length_of_collections = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## embeddings
    emb_path = "jhgan/ko-sroberta-multitask"
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        reordering = LongContextReorder()

        # 1: 전처리x
        no_pre = text_splitter.split_text(context)
        for _ in range(len(no_pre)):
            ids.append(str(idx))
            idx += 1

        client1 = chromadb.PersistentClient(
            path="./patent_db_basic",
            settings=chromadb.Settings(allow_reset=True),
        )
        client1.reset()

        collection1 = client1.get_or_create_collection(
            name="table1",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embeddings
        )
        # DB 저장
        collection1.add(
            documents=no_pre,
            ids=ids,
        )
        length_of_collections.append(len(splits))

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt % 100 == 0:
                print(f'현재 {ques_cnt}')
            if ques_cnt % 500 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-1 정확도는 {cor_top1 / (cor_top1 + not_cor_top1)} 입니다.'
                      f'1: 현재 top-2 정확도는 {cor_top2 / (cor_top2 + not_cor_top2)} 입니다.'
                      f'1: 현재 top-3 정확도는 {cor_top3 / (cor_top3 + not_cor_top3)} 입니다. 평균 시간: {times1 / (cor_top1 + not_cor_top1)}\n'
                      f'2: 현재 top-1 정확도는 {cor_re_top1 / (cor_re_top1 + not_cor_re_top1)} 입니다.'
                      f'2: 현재 top-2 정확도는 {cor_re_top2 / (cor_re_top2 + not_cor_re_top2)} 입니다.'
                      f'2: 현재 top-3 정확도는 {cor_re_top3 / (cor_re_top3 + not_cor_re_top3)} 입니다. 평균 시간: {times2 / (cor_re_top3 + not_cor_re_top3)}\n'
                      f'3: 현재 top-1 정확도는 {cor_re_top1_re / (cor_re_top1_re + not_cor_re_top1_re)} 입니다.'
                      f'3: 현재 top-2 정확도는 {cor_re_top2_re / (cor_re_top2_re + not_cor_re_top2_re)} 입니다.'
                      f'3: 현재 top-3 정확도는 {cor_re_top3_re / (cor_re_top3_re + not_cor_re_top3_re)} 입니다. 평균 시간: {times3 / (cor_re_top3_re + not_cor_re_top3_re)}\n'
                      f'4: 현재 top-1 정확도는 {cor_top1_re / (cor_top1_re + not_cor_top1_re)} 입니다.'
                      f'4: 현재 top-2 정확도는 {cor_top2_re / (cor_top2_re + not_cor_top2_re)} 입니다.'
                      f'4: 현재 top-3 정확도는 {cor_top3_re / (cor_top3_re + not_cor_top3_re)} 입니다. 평균 시간: {times4 / (cor_top3_re + not_cor_top3_re)}\n'
                      )

            # 1: 바로 topk
            start_time = time.time()
            item1 = collection1.query(
                query_texts=[ques],
                n_results=3,
            )
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            # top-1
            cur_score = eval_em(scheme[:1], answers[i])
            if cur_score == 1:
                cor_top1 += 1
            else:
                not_cor_top1 += 1

            # top-2
            cur_score = eval_em(scheme[:2], answers[i])
            if cur_score == 1:
                cor_top2 += 1
            else:
                not_cor_top2 += 1

            # top-3
            cur_score = eval_em(scheme, answers[i])
            if cur_score == 1:
                cor_top3 += 1
            else:
                not_cor_top3 += 1
            times1 += (time.time() - start_time)


            # 2: 리오더 - topk
            start_time = time.time()
            item2 = collection1.query(
                query_texts=[ques],
                n_results=collection1.count(),
            )
            reordered_docs1 = reordering.transform_documents(item2['documents'])

            selected_list = [reordered_docs1[0][0], reordered_docs1[0][-1], reordered_docs1[0][1]]
            scheme = []
            for v in selected_list:
                scheme.append(v)

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_re_top1 += 1
            else:
                not_cor_re_top1 += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_re_top2 += 1
            else:
                not_cor_re_top2 += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_re_top3 += 1
            else:
                not_cor_re_top3 += 1
            times2 += (time.time() - start_time)

            # 3: re-top-re
            start_time = time.time()
            pairs = []
            for item in selected_list:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_re_top1_re += 1
            else:
                not_cor_re_top1_re += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_re_top2_re += 1
            else:
                not_cor_re_top2_re += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_re_top3_re += 1
            else:
                not_cor_re_top3_re += 1
            times3 += (time.time() - start_time)

            # 4
            start_time = time.time()
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            pairs = []
            for item in scheme:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_top1_re += 1
            else:
                not_cor_top1_re += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_top2_re += 1
            else:
                not_cor_top2_re += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_top3_re += 1
            else:
                not_cor_top3_re += 1
            times4 += (time.time() - start_time)

        if ques_cnt >= 3000:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-1 정확도는 {cor_top1 / (cor_top1 + not_cor_top1)} 입니다.'
          f'1: 현재 top-2 정확도는 {cor_top2 / (cor_top2 + not_cor_top2)} 입니다.'
          f'1: 현재 top-3 정확도는 {cor_top3 / (cor_top3 + not_cor_top3)} 입니다. 평균 시간: {times1 / (cor_top1 + not_cor_top1)}\n'
          f'2: 현재 top-1 정확도는 {cor_re_top1 / (cor_re_top1 + not_cor_re_top1)} 입니다.'
          f'2: 현재 top-2 정확도는 {cor_re_top2 / (cor_re_top2 + not_cor_re_top2)} 입니다.'
          f'2: 현재 top-3 정확도는 {cor_re_top3 / (cor_re_top3 + not_cor_re_top3)} 입니다. 평균 시간: {times2 / (cor_re_top3 + not_cor_re_top3)}\n'
          f'3: 현재 top-1 정확도는 {cor_re_top1_re / (cor_re_top1_re + not_cor_re_top1_re)} 입니다.'
          f'3: 현재 top-2 정확도는 {cor_re_top2_re / (cor_re_top2_re + not_cor_re_top2_re)} 입니다.'
          f'3: 현재 top-3 정확도는 {cor_re_top3_re / (cor_re_top3_re + not_cor_re_top3_re)} 입니다. 평균 시간: {times3 / (cor_re_top3_re + not_cor_re_top3_re)}\n'
          f'4: 현재 top-1 정확도는 {cor_top1_re / (cor_top1_re + not_cor_top1_re)} 입니다.'
          f'4: 현재 top-2 정확도는 {cor_top2_re / (cor_top2_re + not_cor_top2_re)} 입니다.'
          f'4: 현재 top-3 정확도는 {cor_top3_re / (cor_top3_re + not_cor_top3_re)} 입니다. 평균 시간: {times4 / (cor_top3_re + not_cor_top3_re)}\n'
          )
    plt.hist(length_of_collections, bins=1, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()

def table_chunk_model():
    # 필요 변수들
    cnt = 0
    score1, score2, score3, score4 = 0, 0, 0, 0# 총점수 저장
    ques_cnt = 0 #질문 총 개수

    # topk 시 3가지 저장 변수
    cor_top1, not_cor_top1 = 0, 0
    cor_top2, not_cor_top2 = 0, 0
    cor_top3, not_cor_top3 = 0, 0

    # reorder - topk
    cor_re_top1, not_cor_re_top1 = 0, 0
    cor_re_top2, not_cor_re_top2 = 0, 0
    cor_re_top3, not_cor_re_top3 = 0, 0

    # reorder - topk - rerank
    cor_re_top1_re, not_cor_re_top1_re = 0, 0
    cor_re_top2_re, not_cor_re_top2_re = 0, 0
    cor_re_top3_re, not_cor_re_top3_re = 0, 0

    # topk - rerank
    cor_top1_re, not_cor_top1_re = 0, 0
    cor_top2_re, not_cor_top2_re = 0, 0
    cor_top3_re, not_cor_top3_re = 0, 0

    times1, times2, times3, times4 = 0.0 ,0.0, 0.0, 0.0
    length_of_collections = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## embeddings
    emb_path = "jhgan/ko-sroberta-multitask"
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        reordering = LongContextReorder()

        # 1: 전처리x
        table_chunk = context.split('<table>')[1].split('</table')[0]
        table_chunk = text_splitter.split_text(table_chunk)
        for _ in range(len(table_chunk)):
            ids.append(str(idx))
            idx += 1

        client1 = chromadb.PersistentClient(
            path="./patent_db_basic2",
            settings=chromadb.Settings(allow_reset=True),
        )
        client1.reset()

        collection1 = client1.get_or_create_collection(
            name="table2",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embeddings
        )
        # DB 저장
        collection1.add(
            documents=table_chunk,
            ids=ids,
        )
        length_of_collections.append(len(splits))

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt % 100 == 0:
                print(f'현재 {ques_cnt}')
            if ques_cnt % 500 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-1 정확도는 {cor_top1 / (cor_top1 + not_cor_top1)} 입니다.'
                      f'1: 현재 top-2 정확도는 {cor_top2 / (cor_top2 + not_cor_top2)} 입니다.'
                      f'1: 현재 top-3 정확도는 {cor_top3 / (cor_top3 + not_cor_top3)} 입니다. 평균 시간: {times1 / (cor_top1 + not_cor_top1)}\n'
                      f'2: 현재 top-1 정확도는 {cor_re_top1 / (cor_re_top1 + not_cor_re_top1)} 입니다.'
                      f'2: 현재 top-2 정확도는 {cor_re_top2 / (cor_re_top2 + not_cor_re_top2)} 입니다.'
                      f'2: 현재 top-3 정확도는 {cor_re_top3 / (cor_re_top3 + not_cor_re_top3)} 입니다. 평균 시간: {times2 / (cor_re_top3 + not_cor_re_top3)}\n'
                      f'3: 현재 top-1 정확도는 {cor_re_top1_re / (cor_re_top1_re + not_cor_re_top1_re)} 입니다.'
                      f'3: 현재 top-2 정확도는 {cor_re_top2_re / (cor_re_top2_re + not_cor_re_top2_re)} 입니다.'
                      f'3: 현재 top-3 정확도는 {cor_re_top3_re / (cor_re_top3_re + not_cor_re_top3_re)} 입니다. 평균 시간: {times3 / (cor_re_top3_re + not_cor_re_top3_re)}\n'
                      f'4: 현재 top-1 정확도는 {cor_top1_re / (cor_top1_re + not_cor_top1_re)} 입니다.'
                      f'4: 현재 top-2 정확도는 {cor_top2_re / (cor_top2_re + not_cor_top2_re)} 입니다.'
                      f'4: 현재 top-3 정확도는 {cor_top3_re / (cor_top3_re + not_cor_top3_re)} 입니다. 평균 시간: {times4 / (cor_top3_re + not_cor_top3_re)}\n'
                      )

            # 1: 바로 topk
            start_time = time.time()
            item1 = collection1.query(
                query_texts=[ques],
                n_results=3,
            )
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            # top-1
            cur_score = eval_em(scheme[:1], answers[i])
            if cur_score == 1:
                cor_top1 += 1
            else:
                not_cor_top1 += 1

            # top-2
            cur_score = eval_em(scheme[:2], answers[i])
            if cur_score == 1:
                cor_top2 += 1
            else:
                not_cor_top2 += 1

            # top-3
            cur_score = eval_em(scheme, answers[i])
            if cur_score == 1:
                cor_top3 += 1
            else:
                not_cor_top3 += 1
            times1 += (time.time() - start_time)


            # 2: 리오더 - topk
            start_time = time.time()
            item2 = collection1.query(
                query_texts=[ques],
                n_results=collection1.count(),
            )
            reordered_docs1 = reordering.transform_documents(item2['documents'])

            selected_list = [reordered_docs1[0][0], reordered_docs1[0][-1], reordered_docs1[0][1]]
            scheme = []
            for v in selected_list:
                scheme.append(v)

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_re_top1 += 1
            else:
                not_cor_re_top1 += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_re_top2 += 1
            else:
                not_cor_re_top2 += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_re_top3 += 1
            else:
                not_cor_re_top3 += 1
            times2 += (time.time() - start_time)

            # 3: re-top-re
            start_time = time.time()
            pairs = []
            for item in selected_list:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_re_top1_re += 1
            else:
                not_cor_re_top1_re += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_re_top2_re += 1
            else:
                not_cor_re_top2_re += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_re_top3_re += 1
            else:
                not_cor_re_top3_re += 1
            times3 += (time.time() - start_time)

            # 4
            start_time = time.time()
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            pairs = []
            for item in scheme:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_top1_re += 1
            else:
                not_cor_top1_re += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_top2_re += 1
            else:
                not_cor_top2_re += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_top3_re += 1
            else:
                not_cor_top3_re += 1
            times4 += (time.time() - start_time)

        if ques_cnt >= 3000:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-1 정확도는 {cor_top1 / (cor_top1 + not_cor_top1)} 입니다.'
          f'1: 현재 top-2 정확도는 {cor_top2 / (cor_top2 + not_cor_top2)} 입니다.'
          f'1: 현재 top-3 정확도는 {cor_top3 / (cor_top3 + not_cor_top3)} 입니다. 평균 시간: {times1 / (cor_top1 + not_cor_top1)}\n'
          f'2: 현재 top-1 정확도는 {cor_re_top1 / (cor_re_top1 + not_cor_re_top1)} 입니다.'
          f'2: 현재 top-2 정확도는 {cor_re_top2 / (cor_re_top2 + not_cor_re_top2)} 입니다.'
          f'2: 현재 top-3 정확도는 {cor_re_top3 / (cor_re_top3 + not_cor_re_top3)} 입니다. 평균 시간: {times2 / (cor_re_top3 + not_cor_re_top3)}\n'
          f'3: 현재 top-1 정확도는 {cor_re_top1_re / (cor_re_top1_re + not_cor_re_top1_re)} 입니다.'
          f'3: 현재 top-2 정확도는 {cor_re_top2_re / (cor_re_top2_re + not_cor_re_top2_re)} 입니다.'
          f'3: 현재 top-3 정확도는 {cor_re_top3_re / (cor_re_top3_re + not_cor_re_top3_re)} 입니다. 평균 시간: {times3 / (cor_re_top3_re + not_cor_re_top3_re)}\n'
          f'4: 현재 top-1 정확도는 {cor_top1_re / (cor_top1_re + not_cor_top1_re)} 입니다.'
          f'4: 현재 top-2 정확도는 {cor_top2_re / (cor_top2_re + not_cor_top2_re)} 입니다.'
          f'4: 현재 top-3 정확도는 {cor_top3_re / (cor_top3_re + not_cor_top3_re)} 입니다. 평균 시간: {times4 / (cor_top3_re + not_cor_top3_re)}\n'
          )
    plt.hist(length_of_collections, bins=1, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()

def table_split_model():
    # 필요 변수들
    cnt = 0
    score1, score2, score3, score4 = 0, 0, 0, 0# 총점수 저장
    ques_cnt = 0 #질문 총 개수

    # topk 시 3가지 저장 변수
    cor_top1, not_cor_top1 = 0, 0
    cor_top2, not_cor_top2 = 0, 0
    cor_top3, not_cor_top3 = 0, 0

    # reorder - topk
    cor_re_top1, not_cor_re_top1 = 0, 0
    cor_re_top2, not_cor_re_top2 = 0, 0
    cor_re_top3, not_cor_re_top3 = 0, 0

    # reorder - topk - rerank
    cor_re_top1_re, not_cor_re_top1_re = 0, 0
    cor_re_top2_re, not_cor_re_top2_re = 0, 0
    cor_re_top3_re, not_cor_re_top3_re = 0, 0

    # topk - rerank
    cor_top1_re, not_cor_top1_re = 0, 0
    cor_top2_re, not_cor_top2_re = 0, 0
    cor_top3_re, not_cor_top3_re = 0, 0

    times1, times2, times3, times4 = 0.0 ,0.0, 0.0, 0.0
    length_of_collections = []

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## embeddings
    emb_path = "jhgan/ko-sroberta-multitask"
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        reordering = LongContextReorder()

        # 1: 전처리x
        table_chunk = HTML_Processor.get_table_data(context)
        table_chunk = table_to_format(table_chunk)
        for _ in range(len(table_chunk)):
            ids.append(str(idx))
            idx += 1

        client1 = chromadb.PersistentClient(
            path="./patent_db_basic4",
            settings=chromadb.Settings(allow_reset=True),
        )
        client1.reset()

        collection1 = client1.get_or_create_collection(
            name="table4",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embeddings
        )
        # DB 저장
        collection1.add(
            documents=table_chunk,
            ids=ids,
        )
        length_of_collections.append(len(splits))

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt % 100 == 0:
                print(f'현재 {ques_cnt}')
            if ques_cnt % 500 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-1 정확도는 {cor_top1 / (cor_top1 + not_cor_top1)} 입니다.'
                      f'1: 현재 top-2 정확도는 {cor_top2 / (cor_top2 + not_cor_top2)} 입니다.'
                      f'1: 현재 top-3 정확도는 {cor_top3 / (cor_top3 + not_cor_top3)} 입니다. 평균 시간: {times1 / (cor_top1 + not_cor_top1)}\n'
                      f'2: 현재 top-1 정확도는 {cor_re_top1 / (cor_re_top1 + not_cor_re_top1)} 입니다.'
                      f'2: 현재 top-2 정확도는 {cor_re_top2 / (cor_re_top2 + not_cor_re_top2)} 입니다.'
                      f'2: 현재 top-3 정확도는 {cor_re_top3 / (cor_re_top3 + not_cor_re_top3)} 입니다. 평균 시간: {times2 / (cor_re_top3 + not_cor_re_top3)}\n'
                      f'3: 현재 top-1 정확도는 {cor_re_top1_re / (cor_re_top1_re + not_cor_re_top1_re)} 입니다.'
                      f'3: 현재 top-2 정확도는 {cor_re_top2_re / (cor_re_top2_re + not_cor_re_top2_re)} 입니다.'
                      f'3: 현재 top-3 정확도는 {cor_re_top3_re / (cor_re_top3_re + not_cor_re_top3_re)} 입니다. 평균 시간: {times3 / (cor_re_top3_re + not_cor_re_top3_re)}\n'
                      f'4: 현재 top-1 정확도는 {cor_top1_re / (cor_top1_re + not_cor_top1_re)} 입니다.'
                      f'4: 현재 top-2 정확도는 {cor_top2_re / (cor_top2_re + not_cor_top2_re)} 입니다.'
                      f'4: 현재 top-3 정확도는 {cor_top3_re / (cor_top3_re + not_cor_top3_re)} 입니다. 평균 시간: {times4 / (cor_top3_re + not_cor_top3_re)}\n'
                      )

            # 1: 바로 topk
            start_time = time.time()
            item1 = collection1.query(
                query_texts=[ques],
                n_results=3,
            )
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            # top-1
            cur_score = eval_em(scheme[:1], answers[i])
            if cur_score == 1:
                cor_top1 += 1
            else:
                not_cor_top1 += 1

            # top-2
            cur_score = eval_em(scheme[:2], answers[i])
            if cur_score == 1:
                cor_top2 += 1
            else:
                not_cor_top2 += 1

            # top-3
            cur_score = eval_em(scheme, answers[i])
            if cur_score == 1:
                cor_top3 += 1
            else:
                not_cor_top3 += 1
            times1 += (time.time() - start_time)


            # 2: 리오더 - topk
            start_time = time.time()
            item2 = collection1.query(
                query_texts=[ques],
                n_results=collection1.count(),
            )
            reordered_docs1 = reordering.transform_documents(item2['documents'])

            selected_list = [reordered_docs1[0][0], reordered_docs1[0][-1], reordered_docs1[0][1]]
            scheme = []
            for v in selected_list:
                scheme.append(v)

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_re_top1 += 1
            else:
                not_cor_re_top1 += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_re_top2 += 1
            else:
                not_cor_re_top2 += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_re_top3 += 1
            else:
                not_cor_re_top3 += 1
            times2 += (time.time() - start_time)

            # 3: re-top-re
            start_time = time.time()
            pairs = []
            for item in selected_list:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_re_top1_re += 1
            else:
                not_cor_re_top1_re += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_re_top2_re += 1
            else:
                not_cor_re_top2_re += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_re_top3_re += 1
            else:
                not_cor_re_top3_re += 1
            times3 += (time.time() - start_time)

            # 4
            start_time = time.time()
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            pairs = []
            for item in scheme:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_top1_re += 1
            else:
                not_cor_top1_re += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_top2_re += 1
            else:
                not_cor_top2_re += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_top3_re += 1
            else:
                not_cor_top3_re += 1
            times4 += (time.time() - start_time)

        if ques_cnt >= 3000:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-1 정확도는 {cor_top1 / (cor_top1 + not_cor_top1)} 입니다.'
          f'1: 현재 top-2 정확도는 {cor_top2 / (cor_top2 + not_cor_top2)} 입니다.'
          f'1: 현재 top-3 정확도는 {cor_top3 / (cor_top3 + not_cor_top3)} 입니다. 평균 시간: {times1 / (cor_top1 + not_cor_top1)}\n'
          f'2: 현재 top-1 정확도는 {cor_re_top1 / (cor_re_top1 + not_cor_re_top1)} 입니다.'
          f'2: 현재 top-2 정확도는 {cor_re_top2 / (cor_re_top2 + not_cor_re_top2)} 입니다.'
          f'2: 현재 top-3 정확도는 {cor_re_top3 / (cor_re_top3 + not_cor_re_top3)} 입니다. 평균 시간: {times2 / (cor_re_top3 + not_cor_re_top3)}\n'
          f'3: 현재 top-1 정확도는 {cor_re_top1_re / (cor_re_top1_re + not_cor_re_top1_re)} 입니다.'
          f'3: 현재 top-2 정확도는 {cor_re_top2_re / (cor_re_top2_re + not_cor_re_top2_re)} 입니다.'
          f'3: 현재 top-3 정확도는 {cor_re_top3_re / (cor_re_top3_re + not_cor_re_top3_re)} 입니다. 평균 시간: {times3 / (cor_re_top3_re + not_cor_re_top3_re)}\n'
          f'4: 현재 top-1 정확도는 {cor_top1_re / (cor_top1_re + not_cor_top1_re)} 입니다.'
          f'4: 현재 top-2 정확도는 {cor_top2_re / (cor_top2_re + not_cor_top2_re)} 입니다.'
          f'4: 현재 top-3 정확도는 {cor_top3_re / (cor_top3_re + not_cor_top3_re)} 입니다. 평균 시간: {times4 / (cor_top3_re + not_cor_top3_re)}\n'
          )
    plt.hist(length_of_collections, bins=len(length_of_collections)-10, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()

def table_row_model():
    # 필요 변수들
    cnt = 0
    score1, score2, score3, score4 = 0, 0, 0, 0# 총점수 저장
    ques_cnt = 0 #질문 총 개수

    # topk 시 3가지 저장 변수
    cor_top1, not_cor_top1 = 0, 0
    cor_top2, not_cor_top2 = 0, 0
    cor_top3, not_cor_top3 = 0, 0

    # reorder - topk
    cor_re_top1, not_cor_re_top1 = 0, 0
    cor_re_top2, not_cor_re_top2 = 0, 0
    cor_re_top3, not_cor_re_top3 = 0, 0

    # reorder - topk - rerank
    cor_re_top1_re, not_cor_re_top1_re = 0, 0
    cor_re_top2_re, not_cor_re_top2_re = 0, 0
    cor_re_top3_re, not_cor_re_top3_re = 0, 0

    # topk - rerank
    cor_top1_re, not_cor_top1_re = 0, 0
    cor_top2_re, not_cor_top2_re = 0, 0
    cor_top3_re, not_cor_top3_re = 0, 0

    times1, times2, times3, times4 = 0.0 ,0.0, 0.0, 0.0
    length_of_collections = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## embeddings
    emb_path = "jhgan/ko-sroberta-multitask"
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        reordering = LongContextReorder()

        # 1: 전처리x
        table_chunk = context.split('<table>')[1].split('</table')[0]
        table_chunk = table_chunk.split('</td></tr>')
        for _ in range(len(table_chunk)):
            ids.append(str(idx))
            idx += 1

        client1 = chromadb.PersistentClient(
            path="./patent_db_basic2",
            settings=chromadb.Settings(allow_reset=True),
        )
        client1.reset()

        collection1 = client1.get_or_create_collection(
            name="table2",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embeddings
        )
        # DB 저장
        collection1.add(
            documents=table_chunk,
            ids=ids,
        )
        length_of_collections.append(len(splits))

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt % 100 == 0:
                print(f'현재 {ques_cnt}')
            if ques_cnt % 500 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-1 정확도는 {cor_top1 / (cor_top1 + not_cor_top1)} 입니다.'
                      f'1: 현재 top-2 정확도는 {cor_top2 / (cor_top2 + not_cor_top2)} 입니다.'
                      f'1: 현재 top-3 정확도는 {cor_top3 / (cor_top3 + not_cor_top3)} 입니다. 평균 시간: {times1 / (cor_top1 + not_cor_top1)}\n'
                      f'2: 현재 top-1 정확도는 {cor_re_top1 / (cor_re_top1 + not_cor_re_top1)} 입니다.'
                      f'2: 현재 top-2 정확도는 {cor_re_top2 / (cor_re_top2 + not_cor_re_top2)} 입니다.'
                      f'2: 현재 top-3 정확도는 {cor_re_top3 / (cor_re_top3 + not_cor_re_top3)} 입니다. 평균 시간: {times2 / (cor_re_top3 + not_cor_re_top3)}\n'
                      f'3: 현재 top-1 정확도는 {cor_re_top1_re / (cor_re_top1_re + not_cor_re_top1_re)} 입니다.'
                      f'3: 현재 top-2 정확도는 {cor_re_top2_re / (cor_re_top2_re + not_cor_re_top2_re)} 입니다.'
                      f'3: 현재 top-3 정확도는 {cor_re_top3_re / (cor_re_top3_re + not_cor_re_top3_re)} 입니다. 평균 시간: {times3 / (cor_re_top3_re + not_cor_re_top3_re)}\n'
                      f'4: 현재 top-1 정확도는 {cor_top1_re / (cor_top1_re + not_cor_top1_re)} 입니다.'
                      f'4: 현재 top-2 정확도는 {cor_top2_re / (cor_top2_re + not_cor_top2_re)} 입니다.'
                      f'4: 현재 top-3 정확도는 {cor_top3_re / (cor_top3_re + not_cor_top3_re)} 입니다. 평균 시간: {times4 / (cor_top3_re + not_cor_top3_re)}\n'
                      )

            # 1: 바로 topk
            start_time = time.time()
            item1 = collection1.query(
                query_texts=[ques],
                n_results=3,
            )
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            # top-1
            cur_score = eval_em(scheme[:1], answers[i])
            if cur_score == 1:
                cor_top1 += 1
            else:
                not_cor_top1 += 1

            # top-2
            cur_score = eval_em(scheme[:2], answers[i])
            if cur_score == 1:
                cor_top2 += 1
            else:
                not_cor_top2 += 1

            # top-3
            cur_score = eval_em(scheme, answers[i])
            if cur_score == 1:
                cor_top3 += 1
            else:
                not_cor_top3 += 1
            times1 += (time.time() - start_time)


            # 2: 리오더 - topk
            start_time = time.time()
            item2 = collection1.query(
                query_texts=[ques],
                n_results=collection1.count(),
            )
            reordered_docs1 = reordering.transform_documents(item2['documents'])

            selected_list = [reordered_docs1[0][0], reordered_docs1[0][-1], reordered_docs1[0][1]]
            scheme = []
            for v in selected_list:
                scheme.append(v)

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_re_top1 += 1
            else:
                not_cor_re_top1 += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_re_top2 += 1
            else:
                not_cor_re_top2 += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_re_top3 += 1
            else:
                not_cor_re_top3 += 1
            times2 += (time.time() - start_time)

            # 3: re-top-re
            start_time = time.time()
            pairs = []
            for item in selected_list:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_re_top1_re += 1
            else:
                not_cor_re_top1_re += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_re_top2_re += 1
            else:
                not_cor_re_top2_re += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_re_top3_re += 1
            else:
                not_cor_re_top3_re += 1
            times3 += (time.time() - start_time)

            # 4
            start_time = time.time()
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            pairs = []
            for item in scheme:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_top1_re += 1
            else:
                not_cor_top1_re += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_top2_re += 1
            else:
                not_cor_top2_re += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_top3_re += 1
            else:
                not_cor_top3_re += 1
            times4 += (time.time() - start_time)

        if ques_cnt >= 3000:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-1 정확도는 {cor_top1 / (cor_top1 + not_cor_top1)} 입니다.'
          f'1: 현재 top-2 정확도는 {cor_top2 / (cor_top2 + not_cor_top2)} 입니다.'
          f'1: 현재 top-3 정확도는 {cor_top3 / (cor_top3 + not_cor_top3)} 입니다. 평균 시간: {times1 / (cor_top1 + not_cor_top1)}\n'
          f'2: 현재 top-1 정확도는 {cor_re_top1 / (cor_re_top1 + not_cor_re_top1)} 입니다.'
          f'2: 현재 top-2 정확도는 {cor_re_top2 / (cor_re_top2 + not_cor_re_top2)} 입니다.'
          f'2: 현재 top-3 정확도는 {cor_re_top3 / (cor_re_top3 + not_cor_re_top3)} 입니다. 평균 시간: {times2 / (cor_re_top3 + not_cor_re_top3)}\n'
          f'3: 현재 top-1 정확도는 {cor_re_top1_re / (cor_re_top1_re + not_cor_re_top1_re)} 입니다.'
          f'3: 현재 top-2 정확도는 {cor_re_top2_re / (cor_re_top2_re + not_cor_re_top2_re)} 입니다.'
          f'3: 현재 top-3 정확도는 {cor_re_top3_re / (cor_re_top3_re + not_cor_re_top3_re)} 입니다. 평균 시간: {times3 / (cor_re_top3_re + not_cor_re_top3_re)}\n'
          f'4: 현재 top-1 정확도는 {cor_top1_re / (cor_top1_re + not_cor_top1_re)} 입니다.'
          f'4: 현재 top-2 정확도는 {cor_top2_re / (cor_top2_re + not_cor_top2_re)} 입니다.'
          f'4: 현재 top-3 정확도는 {cor_top3_re / (cor_top3_re + not_cor_top3_re)} 입니다. 평균 시간: {times4 / (cor_top3_re + not_cor_top3_re)}\n'
          )
    plt.hist(length_of_collections, bins=len(length_of_collections)-10, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()

def table_serialization_model():
    # 필요 변수들
    cnt = 0
    score1, score2, score3, score4 = 0, 0, 0, 0# 총점수 저장
    ques_cnt = 0 #질문 총 개수

    # topk 시 3가지 저장 변수
    cor_top1, not_cor_top1 = 0, 0
    cor_top2, not_cor_top2 = 0, 0
    cor_top3, not_cor_top3 = 0, 0

    # reorder - topk
    cor_re_top1, not_cor_re_top1 = 0, 0
    cor_re_top2, not_cor_re_top2 = 0, 0
    cor_re_top3, not_cor_re_top3 = 0, 0

    # reorder - topk - rerank
    cor_re_top1_re, not_cor_re_top1_re = 0, 0
    cor_re_top2_re, not_cor_re_top2_re = 0, 0
    cor_re_top3_re, not_cor_re_top3_re = 0, 0

    # topk - rerank
    cor_top1_re, not_cor_top1_re = 0, 0
    cor_top2_re, not_cor_top2_re = 0, 0
    cor_top3_re, not_cor_top3_re = 0, 0

    times1, times2, times3, times4 = 0.0 ,0.0, 0.0, 0.0
    length_of_collections = []

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## embeddings
    emb_path = "jhgan/ko-sroberta-multitask"
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        reordering = LongContextReorder()

        # 1: 전처리x
        table_chunk = HTML_Processor.get_table_data(context)
        table_chunk = table_to_sentence(table_chunk)
        for _ in range(len(table_chunk)):
            ids.append(str(idx))
            idx += 1

        client1 = chromadb.PersistentClient(
            path="./patent_db_basic5",
            settings=chromadb.Settings(allow_reset=True),
        )
        client1.reset()

        collection1 = client1.get_or_create_collection(
            name="table5",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embeddings
        )
        # DB 저장
        collection1.add(
            documents=table_chunk,
            ids=ids,
        )
        length_of_collections.append(len(splits))

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt % 100 == 0:
                print(f'현재 {ques_cnt}')
            if ques_cnt % 500 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-1 정확도는 {cor_top1 / (cor_top1 + not_cor_top1)} 입니다.'
                      f'1: 현재 top-2 정확도는 {cor_top2 / (cor_top2 + not_cor_top2)} 입니다.'
                      f'1: 현재 top-3 정확도는 {cor_top3 / (cor_top3 + not_cor_top3)} 입니다. 평균 시간: {times1 / (cor_top1 + not_cor_top1)}\n'
                      f'2: 현재 top-1 정확도는 {cor_re_top1 / (cor_re_top1 + not_cor_re_top1)} 입니다.'
                      f'2: 현재 top-2 정확도는 {cor_re_top2 / (cor_re_top2 + not_cor_re_top2)} 입니다.'
                      f'2: 현재 top-3 정확도는 {cor_re_top3 / (cor_re_top3 + not_cor_re_top3)} 입니다. 평균 시간: {times2 / (cor_re_top3 + not_cor_re_top3)}\n'
                      f'3: 현재 top-1 정확도는 {cor_re_top1_re / (cor_re_top1_re + not_cor_re_top1_re)} 입니다.'
                      f'3: 현재 top-2 정확도는 {cor_re_top2_re / (cor_re_top2_re + not_cor_re_top2_re)} 입니다.'
                      f'3: 현재 top-3 정확도는 {cor_re_top3_re / (cor_re_top3_re + not_cor_re_top3_re)} 입니다. 평균 시간: {times3 / (cor_re_top3_re + not_cor_re_top3_re)}\n'
                      f'4: 현재 top-1 정확도는 {cor_top1_re / (cor_top1_re + not_cor_top1_re)} 입니다.'
                      f'4: 현재 top-2 정확도는 {cor_top2_re / (cor_top2_re + not_cor_top2_re)} 입니다.'
                      f'4: 현재 top-3 정확도는 {cor_top3_re / (cor_top3_re + not_cor_top3_re)} 입니다. 평균 시간: {times4 / (cor_top3_re + not_cor_top3_re)}\n'
                      )

            # 1: 바로 topk
            start_time = time.time()
            item1 = collection1.query(
                query_texts=[ques],
                n_results=3,
            )
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            # top-1
            cur_score = eval_em(scheme[:1], answers[i])
            if cur_score == 1:
                cor_top1 += 1
            else:
                not_cor_top1 += 1

            # top-2
            cur_score = eval_em(scheme[:2], answers[i])
            if cur_score == 1:
                cor_top2 += 1
            else:
                not_cor_top2 += 1

            # top-3
            cur_score = eval_em(scheme, answers[i])
            if cur_score == 1:
                cor_top3 += 1
            else:
                not_cor_top3 += 1
            times1 += (time.time() - start_time)


            # 2: 리오더 - topk
            start_time = time.time()
            item2 = collection1.query(
                query_texts=[ques],
                n_results=collection1.count(),
            )
            reordered_docs1 = reordering.transform_documents(item2['documents'])

            selected_list = [reordered_docs1[0][0], reordered_docs1[0][-1], reordered_docs1[0][1]]
            scheme = []
            for v in selected_list:
                scheme.append(v)

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_re_top1 += 1
            else:
                not_cor_re_top1 += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_re_top2 += 1
            else:
                not_cor_re_top2 += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_re_top3 += 1
            else:
                not_cor_re_top3 += 1
            times2 += (time.time() - start_time)

            # 3: re-top-re
            start_time = time.time()
            pairs = []
            for item in selected_list:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_re_top1_re += 1
            else:
                not_cor_re_top1_re += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_re_top2_re += 1
            else:
                not_cor_re_top2_re += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_re_top3_re += 1
            else:
                not_cor_re_top3_re += 1
            times3 += (time.time() - start_time)

            # 4
            start_time = time.time()
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            pairs = []
            for item in scheme:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            cur_score2 = eval_em(scheme[:1], answers[i])
            if cur_score2 == 1:
                cor_top1_re += 1
            else:
                not_cor_top1_re += 1

            # top-2
            cur_score2 = eval_em(scheme[:2], answers[i])
            if cur_score2 == 1:
                cor_top2_re += 1
            else:
                not_cor_top2_re += 1

            # top-3
            cur_score2 = eval_em(scheme, answers[i])
            if cur_score2 == 1:
                cor_top3_re += 1
            else:
                not_cor_top3_re += 1
            times4 += (time.time() - start_time)

        if ques_cnt >= 3000:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-1 정확도는 {cor_top1 / (cor_top1 + not_cor_top1)} 입니다.'
          f'1: 현재 top-2 정확도는 {cor_top2 / (cor_top2 + not_cor_top2)} 입니다.'
          f'1: 현재 top-3 정확도는 {cor_top3 / (cor_top3 + not_cor_top3)} 입니다. 평균 시간: {times1 / (cor_top1 + not_cor_top1)}\n'
          f'2: 현재 top-1 정확도는 {cor_re_top1 / (cor_re_top1 + not_cor_re_top1)} 입니다.'
          f'2: 현재 top-2 정확도는 {cor_re_top2 / (cor_re_top2 + not_cor_re_top2)} 입니다.'
          f'2: 현재 top-3 정확도는 {cor_re_top3 / (cor_re_top3 + not_cor_re_top3)} 입니다. 평균 시간: {times2 / (cor_re_top3 + not_cor_re_top3)}\n'
          f'3: 현재 top-1 정확도는 {cor_re_top1_re / (cor_re_top1_re + not_cor_re_top1_re)} 입니다.'
          f'3: 현재 top-2 정확도는 {cor_re_top2_re / (cor_re_top2_re + not_cor_re_top2_re)} 입니다.'
          f'3: 현재 top-3 정확도는 {cor_re_top3_re / (cor_re_top3_re + not_cor_re_top3_re)} 입니다. 평균 시간: {times3 / (cor_re_top3_re + not_cor_re_top3_re)}\n'
          f'4: 현재 top-1 정확도는 {cor_top1_re / (cor_top1_re + not_cor_top1_re)} 입니다.'
          f'4: 현재 top-2 정확도는 {cor_top2_re / (cor_top2_re + not_cor_top2_re)} 입니다.'
          f'4: 현재 top-3 정확도는 {cor_top3_re / (cor_top3_re + not_cor_top3_re)} 입니다. 평균 시간: {times4 / (cor_top3_re + not_cor_top3_re)}\n'
          )
    plt.hist(length_of_collections, bins=len(length_of_collections)-10, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()


## OpenAI 질의응답 모델
openai.api_key = 'api-key'
def test_zero_shot_model():
    # 필요 변수들
    score1_1, score1_2, score1_3 = 0, 0, 0# 총점수 저장
    ques_cnt = 0 #질문 총 개수
    times1 = 0.0
    length_of_collections = []

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt % 10 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-3 정확도는 {score1_1 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
                      )

            start_time = time.time()
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 만약 질문에 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    # {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score1_1 += f1_score(gpt_ans1, answers[i])
            times1 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

        if ques_cnt >= 100:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-3 정확도는 {score1_1 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
          )
    plt.hist(length_of_collections, bins=len(length_of_collections)-10, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()

def test_no_pre_model():
    # 필요 변수들
    score1_1, score1_2, score1_3 = 0, 0, 0 # 총점수 저장
    score2_1, score2_2, score2_3 = 0, 0, 0  # 총점수 저장
    score3_1, score3_2, score3_3 = 0, 0, 0  # 총점수 저장
    score4_1, score4_2, score4_3 = 0, 0, 0  # 총점수 저장

    ques_cnt = 0 #질문 총 개수
    times1, times2, times3, times4 = 0.0, 0.0, 0.0, 0.0
    length_of_collections = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## embeddings
    emb_path = "jhgan/ko-sroberta-multitask"
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        reordering = LongContextReorder()

        # 1: 전처리x
        no_pre = text_splitter.split_text(context)

        for _ in range(len(no_pre)):
            ids.append(str(idx))
            idx += 1

        client1 = chromadb.PersistentClient(
            path="./patent_db_basic",
            settings=chromadb.Settings(allow_reset=True),
        )
        client1.reset()

        collection1 = client1.get_or_create_collection(
            name="table1",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embeddings
        )
        # DB 저장
        collection1.add(
            documents=no_pre,
            ids=ids,
        )
        length_of_collections.append(len(splits))

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt % 10 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-1 정확도는 {score1_1 / ques_cnt} 입니다.'
                      f'1: 현재 top-2 정확도는 {score1_2 / ques_cnt} 입니다.'
                      f'1: 현재 top-3 정확도는 {score1_3 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
                      f'2: 현재 top-1 정확도는 {score2_1 / ques_cnt} 입니다.'
                      f'2: 현재 top-2 정확도는 {score2_2 / ques_cnt} 입니다.'
                      f'2: 현재 top-3 정확도는 {score2_3 / ques_cnt} 입니다. 평균 시간: {times2 / (ques_cnt)}\n'
                      f'3: 현재 top-1 정확도는 {score3_1 / ques_cnt} 입니다.'
                      f'3: 현재 top-2 정확도는 {score3_2 / ques_cnt} 입니다.'
                      f'3: 현재 top-3 정확도는 {score3_3 / ques_cnt} 입니다. 평균 시간: {times3 / (ques_cnt)}\n'
                      f'4: 현재 top-1 정확도는 {score4_1 / ques_cnt} 입니다.'
                      f'4: 현재 top-2 정확도는 {score4_2 / ques_cnt} 입니다.'
                      f'4: 현재 top-3 정확도는 {score4_3 / ques_cnt} 입니다. 평균 시간: {times4 / (ques_cnt)}\n'
                      )

            # 1: 바로 topk
            start_time = time.time()
            item1 = collection1.query(
                query_texts=[ques],
                n_results=3,
            )
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score1_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score1_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score1_3 += f1_score(gpt_ans3, answers[i])
            times1 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 2: 리오더 - topk
            start_time = time.time()
            item2 = collection1.query(
                query_texts=[ques],
                n_results=collection1.count(),
            )
            reordered_docs1 = reordering.transform_documents(item2['documents'])

            selected_list = [reordered_docs1[0][0], reordered_docs1[0][-1], reordered_docs1[0][1]]
            scheme = []
            for v in selected_list:
                scheme.append(v)

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score2_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score2_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score2_3 += f1_score(gpt_ans3, answers[i])
            times2 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 3: re-top-re
            start_time = time.time()
            pairs = []
            for item in selected_list:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score3_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score3_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score3_3 += f1_score(gpt_ans3, answers[i])
            times3 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 4
            start_time = time.time()
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            pairs = []
            for item in scheme:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score4_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score4_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score4_3 += f1_score(gpt_ans3, answers[i])
            times4 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

        if ques_cnt >= 100:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-1 정확도는 {score1_1 / ques_cnt} 입니다.'
          f'1: 현재 top-2 정확도는 {score1_2 / ques_cnt} 입니다.'
          f'1: 현재 top-3 정확도는 {score1_3 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
          f'2: 현재 top-1 정확도는 {score2_1 / ques_cnt} 입니다.'
          f'2: 현재 top-2 정확도는 {score2_2 / ques_cnt} 입니다.'
          f'2: 현재 top-3 정확도는 {score2_3 / ques_cnt} 입니다. 평균 시간: {times2 / (ques_cnt)}\n'
          f'3: 현재 top-1 정확도는 {score3_1 / ques_cnt} 입니다.'
          f'3: 현재 top-2 정확도는 {score3_2 / ques_cnt} 입니다.'
          f'3: 현재 top-3 정확도는 {score3_3 / ques_cnt} 입니다. 평균 시간: {times3 / (ques_cnt)}\n'
          f'4: 현재 top-1 정확도는 {score4_1 / ques_cnt} 입니다.'
          f'4: 현재 top-2 정확도는 {score4_2 / ques_cnt} 입니다.'
          f'4: 현재 top-3 정확도는 {score4_3 / ques_cnt} 입니다. 평균 시간: {times4 / (ques_cnt)}\n'
          )
    plt.hist(length_of_collections, bins=1, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()

def test_table_chunk_model():
    # 필요 변수들
    score1_1, score1_2, score1_3 = 0, 0, 0  # 총점수 저장
    score2_1, score2_2, score2_3 = 0, 0, 0  # 총점수 저장
    score3_1, score3_2, score3_3 = 0, 0, 0  # 총점수 저장
    score4_1, score4_2, score4_3 = 0, 0, 0  # 총점수 저장

    ques_cnt = 0  # 질문 총 개수
    times1, times2, times3, times4 = 0.0, 0.0, 0.0, 0.0
    length_of_collections = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## embeddings
    emb_path = "jhgan/ko-sroberta-multitask"
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        reordering = LongContextReorder()

        # 1: 전처리x
        table_chunk = context.split('<table>')[1].split('</table')[0]
        table_chunk = text_splitter.split_text(table_chunk)

        for _ in range(len(table_chunk)):
            ids.append(str(idx))
            idx += 1

        client1 = chromadb.PersistentClient(
            path="./patent_db_test2",
            settings=chromadb.Settings(allow_reset=True),
        )
        client1.reset()

        collection1 = client1.get_or_create_collection(
            name="table22",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embeddings
        )
        # DB 저장
        collection1.add(
            documents=table_chunk,
            ids=ids,
        )
        length_of_collections.append(len(splits))

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt % 10 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-1 정확도는 {score1_1 / ques_cnt} 입니다.'
                      f'1: 현재 top-2 정확도는 {score1_2 / ques_cnt} 입니다.'
                      f'1: 현재 top-3 정확도는 {score1_3 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
                      f'2: 현재 top-1 정확도는 {score2_1 / ques_cnt} 입니다.'
                      f'2: 현재 top-2 정확도는 {score2_2 / ques_cnt} 입니다.'
                      f'2: 현재 top-3 정확도는 {score2_3 / ques_cnt} 입니다. 평균 시간: {times2 / (ques_cnt)}\n'
                      f'3: 현재 top-1 정확도는 {score3_1 / ques_cnt} 입니다.'
                      f'3: 현재 top-2 정확도는 {score3_2 / ques_cnt} 입니다.'
                      f'3: 현재 top-3 정확도는 {score3_3 / ques_cnt} 입니다. 평균 시간: {times3 / (ques_cnt)}\n'
                      f'4: 현재 top-1 정확도는 {score4_1 / ques_cnt} 입니다.'
                      f'4: 현재 top-2 정확도는 {score4_2 / ques_cnt} 입니다.'
                      f'4: 현재 top-3 정확도는 {score4_3 / ques_cnt} 입니다. 평균 시간: {times4 / (ques_cnt)}\n'
                      )

            # 1: 바로 topk
            start_time = time.time()
            item1 = collection1.query(
                query_texts=[ques],
                n_results=3,
            )
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score1_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score1_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score1_3 += f1_score(gpt_ans3, answers[i])
            times1 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 2: 리오더 - topk
            start_time = time.time()
            item2 = collection1.query(
                query_texts=[ques],
                n_results=collection1.count(),
            )
            reordered_docs1 = reordering.transform_documents(item2['documents'])

            selected_list = [reordered_docs1[0][0], reordered_docs1[0][-1], reordered_docs1[0][1]]
            scheme = []
            for v in selected_list:
                scheme.append(v)

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score2_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score2_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score2_3 += f1_score(gpt_ans3, answers[i])
            times2 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 3: re-top-re
            start_time = time.time()
            pairs = []
            for item in selected_list:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score3_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score3_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score3_3 += f1_score(gpt_ans3, answers[i])
            times3 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 4
            start_time = time.time()
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            pairs = []
            for item in scheme:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score4_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score4_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score4_3 += f1_score(gpt_ans3, answers[i])
            times4 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

        if ques_cnt >= 100:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-1 정확도는 {score1_1 / ques_cnt} 입니다.'
          f'1: 현재 top-2 정확도는 {score1_2 / ques_cnt} 입니다.'
          f'1: 현재 top-3 정확도는 {score1_3 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
          f'2: 현재 top-1 정확도는 {score2_1 / ques_cnt} 입니다.'
          f'2: 현재 top-2 정확도는 {score2_2 / ques_cnt} 입니다.'
          f'2: 현재 top-3 정확도는 {score2_3 / ques_cnt} 입니다. 평균 시간: {times2 / (ques_cnt)}\n'
          f'3: 현재 top-1 정확도는 {score3_1 / ques_cnt} 입니다.'
          f'3: 현재 top-2 정확도는 {score3_2 / ques_cnt} 입니다.'
          f'3: 현재 top-3 정확도는 {score3_3 / ques_cnt} 입니다. 평균 시간: {times3 / (ques_cnt)}\n'
          f'4: 현재 top-1 정확도는 {score4_1 / ques_cnt} 입니다.'
          f'4: 현재 top-2 정확도는 {score4_2 / ques_cnt} 입니다.'
          f'4: 현재 top-3 정확도는 {score4_3 / ques_cnt} 입니다. 평균 시간: {times4 / (ques_cnt)}\n'
          )
    plt.hist(length_of_collections, bins=len(length_of_collections)-10, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()

def test_table_row_model():
    # 필요 변수들
    score1_1, score1_2, score1_3 = 0, 0, 0  # 총점수 저장
    score2_1, score2_2, score2_3 = 0, 0, 0  # 총점수 저장
    score3_1, score3_2, score3_3 = 0, 0, 0  # 총점수 저장
    score4_1, score4_2, score4_3 = 0, 0, 0  # 총점수 저장

    ques_cnt = 0  # 질문 총 개수
    times1, times2, times3, times4 = 0.0, 0.0, 0.0, 0.0
    length_of_collections = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## embeddings
    emb_path = "jhgan/ko-sroberta-multitask"
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        reordering = LongContextReorder()

        # 1: 전처리x
        table_chunk = context.split('<table>')[1].split('</table')[0]
        table_chunk = table_chunk.split('</td></tr>')

        for _ in range(len(table_chunk)):
            ids.append(str(idx))
            idx += 1

        client1 = chromadb.PersistentClient(
            path="./patent_db_test3",
            settings=chromadb.Settings(allow_reset=True),
        )
        client1.reset()

        collection1 = client1.get_or_create_collection(
            name="table33",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embeddings
        )
        # DB 저장
        collection1.add(
            documents=table_chunk,
            ids=ids,
        )
        length_of_collections.append(len(splits))

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt < 81: continue
            if ques_cnt % 10 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-1 정확도는 {score1_1 / ques_cnt} 입니다.'
                      f'1: 현재 top-2 정확도는 {score1_2 / ques_cnt} 입니다.'
                      f'1: 현재 top-3 정확도는 {score1_3 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
                      f'2: 현재 top-1 정확도는 {score2_1 / ques_cnt} 입니다.'
                      f'2: 현재 top-2 정확도는 {score2_2 / ques_cnt} 입니다.'
                      f'2: 현재 top-3 정확도는 {score2_3 / ques_cnt} 입니다. 평균 시간: {times2 / (ques_cnt)}\n'
                      f'3: 현재 top-1 정확도는 {score3_1 / ques_cnt} 입니다.'
                      f'3: 현재 top-2 정확도는 {score3_2 / ques_cnt} 입니다.'
                      f'3: 현재 top-3 정확도는 {score3_3 / ques_cnt} 입니다. 평균 시간: {times3 / (ques_cnt)}\n'
                      f'4: 현재 top-1 정확도는 {score4_1 / ques_cnt} 입니다.'
                      f'4: 현재 top-2 정확도는 {score4_2 / ques_cnt} 입니다.'
                      f'4: 현재 top-3 정확도는 {score4_3 / ques_cnt} 입니다. 평균 시간: {times4 / (ques_cnt)}\n'
                      )

            # 1: 바로 topk
            start_time = time.time()
            item1 = collection1.query(
                query_texts=[ques],
                n_results=3,
            )
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score1_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score1_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score1_3 += f1_score(gpt_ans3, answers[i])
            times1 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 2: 리오더 - topk
            start_time = time.time()
            item2 = collection1.query(
                query_texts=[ques],
                n_results=collection1.count(),
            )
            reordered_docs1 = reordering.transform_documents(item2['documents'])

            selected_list = [reordered_docs1[0][0], reordered_docs1[0][-1], reordered_docs1[0][1]]
            scheme = []
            for v in selected_list:
                scheme.append(v)

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score2_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score2_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score2_3 += f1_score(gpt_ans3, answers[i])
            times2 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 3: re-top-re
            start_time = time.time()
            pairs = []
            for item in selected_list:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score3_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score3_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score3_3 += f1_score(gpt_ans3, answers[i])
            times3 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 4
            start_time = time.time()
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            pairs = []
            for item in scheme:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score4_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score4_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score4_3 += f1_score(gpt_ans3, answers[i])
            times4 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

        if ques_cnt >= 100:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-1 정확도는 {score1_1 / ques_cnt} 입니다.'
          f'1: 현재 top-2 정확도는 {score1_2 / ques_cnt} 입니다.'
          f'1: 현재 top-3 정확도는 {score1_3 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
          f'2: 현재 top-1 정확도는 {score2_1 / ques_cnt} 입니다.'
          f'2: 현재 top-2 정확도는 {score2_2 / ques_cnt} 입니다.'
          f'2: 현재 top-3 정확도는 {score2_3 / ques_cnt} 입니다. 평균 시간: {times2 / (ques_cnt)}\n'
          f'3: 현재 top-1 정확도는 {score3_1 / ques_cnt} 입니다.'
          f'3: 현재 top-2 정확도는 {score3_2 / ques_cnt} 입니다.'
          f'3: 현재 top-3 정확도는 {score3_3 / ques_cnt} 입니다. 평균 시간: {times3 / (ques_cnt)}\n'
          f'4: 현재 top-1 정확도는 {score4_1 / ques_cnt} 입니다.'
          f'4: 현재 top-2 정확도는 {score4_2 / ques_cnt} 입니다.'
          f'4: 현재 top-3 정확도는 {score4_3 / ques_cnt} 입니다. 평균 시간: {times4 / (ques_cnt)}\n'
          )
    plt.hist(length_of_collections, bins=len(length_of_collections)-10, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()

def test_table_split_model():
    # 필요 변수들
    score1_1, score1_2, score1_3 = 0, 0, 0  # 총점수 저장
    score2_1, score2_2, score2_3 = 0, 0, 0  # 총점수 저장
    score3_1, score3_2, score3_3 = 0, 0, 0  # 총점수 저장
    score4_1, score4_2, score4_3 = 0, 0, 0  # 총점수 저장

    ques_cnt = 0  # 질문 총 개수
    times1, times2, times3, times4 = 0.0, 0.0, 0.0, 0.0
    length_of_collections = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## embeddings
    emb_path = "jhgan/ko-sroberta-multitask"
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        reordering = LongContextReorder()

        # 1: 전처리x
        table_chunk = HTML_Processor.get_table_data(context)
        table_chunk = table_to_format(table_chunk)

        for _ in range(len(table_chunk)):
            ids.append(str(idx))
            idx += 1

        client1 = chromadb.PersistentClient(
            path="./patent_db_test4",
            settings=chromadb.Settings(allow_reset=True),
        )
        client1.reset()

        collection1 = client1.get_or_create_collection(
            name="table4_",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embeddings
        )
        # DB 저장
        collection1.add(
            documents=table_chunk,
            ids=ids,
        )
        length_of_collections.append(len(splits))

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt % 10 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-1 정확도는 {score1_1 / ques_cnt} 입니다.'
                      f'1: 현재 top-2 정확도는 {score1_2 / ques_cnt} 입니다.'
                      f'1: 현재 top-3 정확도는 {score1_3 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
                      f'2: 현재 top-1 정확도는 {score2_1 / ques_cnt} 입니다.'
                      f'2: 현재 top-2 정확도는 {score2_2 / ques_cnt} 입니다.'
                      f'2: 현재 top-3 정확도는 {score2_3 / ques_cnt} 입니다. 평균 시간: {times2 / (ques_cnt)}\n'
                      f'3: 현재 top-1 정확도는 {score3_1 / ques_cnt} 입니다.'
                      f'3: 현재 top-2 정확도는 {score3_2 / ques_cnt} 입니다.'
                      f'3: 현재 top-3 정확도는 {score3_3 / ques_cnt} 입니다. 평균 시간: {times3 / (ques_cnt)}\n'
                      f'4: 현재 top-1 정확도는 {score4_1 / ques_cnt} 입니다.'
                      f'4: 현재 top-2 정확도는 {score4_2 / ques_cnt} 입니다.'
                      f'4: 현재 top-3 정확도는 {score4_3 / ques_cnt} 입니다. 평균 시간: {times4 / (ques_cnt)}\n'
                      )

            # 1: 바로 topk
            start_time = time.time()
            item1 = collection1.query(
                query_texts=[ques],
                n_results=3,
            )
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score1_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score1_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score1_3 += f1_score(gpt_ans3, answers[i])
            times1 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 2: 리오더 - topk
            start_time = time.time()
            item2 = collection1.query(
                query_texts=[ques],
                n_results=collection1.count(),
            )
            reordered_docs1 = reordering.transform_documents(item2['documents'])

            selected_list = [reordered_docs1[0][0], reordered_docs1[0][-1], reordered_docs1[0][1]]
            scheme = []
            for v in selected_list:
                scheme.append(v)

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score2_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score2_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score2_3 += f1_score(gpt_ans3, answers[i])
            times2 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 3: re-top-re
            start_time = time.time()
            pairs = []
            for item in selected_list:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score3_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score3_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score3_3 += f1_score(gpt_ans3, answers[i])
            times3 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 4
            start_time = time.time()
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            pairs = []
            for item in scheme:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score4_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score4_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score4_3 += f1_score(gpt_ans3, answers[i])
            times4 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

        if ques_cnt >= 100:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-1 정확도는 {score1_1 / ques_cnt} 입니다.'
          f'1: 현재 top-2 정확도는 {score1_2 / ques_cnt} 입니다.'
          f'1: 현재 top-3 정확도는 {score1_3 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
          f'2: 현재 top-1 정확도는 {score2_1 / ques_cnt} 입니다.'
          f'2: 현재 top-2 정확도는 {score2_2 / ques_cnt} 입니다.'
          f'2: 현재 top-3 정확도는 {score2_3 / ques_cnt} 입니다. 평균 시간: {times2 / (ques_cnt)}\n'
          f'3: 현재 top-1 정확도는 {score3_1 / ques_cnt} 입니다.'
          f'3: 현재 top-2 정확도는 {score3_2 / ques_cnt} 입니다.'
          f'3: 현재 top-3 정확도는 {score3_3 / ques_cnt} 입니다. 평균 시간: {times3 / (ques_cnt)}\n'
          f'4: 현재 top-1 정확도는 {score4_1 / ques_cnt} 입니다.'
          f'4: 현재 top-2 정확도는 {score4_2 / ques_cnt} 입니다.'
          f'4: 현재 top-3 정확도는 {score4_3 / ques_cnt} 입니다. 평균 시간: {times4 / (ques_cnt)}\n'
          )
    plt.hist(length_of_collections, bins=len(length_of_collections)-10, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()

def test_table_serialization_model():
    # 필요 변수들
    score1_1, score1_2, score1_3 = 0, 0, 0  # 총점수 저장
    score2_1, score2_2, score2_3 = 0, 0, 0  # 총점수 저장
    score3_1, score3_2, score3_3 = 0, 0, 0  # 총점수 저장
    score4_1, score4_2, score4_3 = 0, 0, 0  # 총점수 저장

    ques_cnt = 0  # 질문 총 개수
    times1, times2, times3, times4 = 0.0, 0.0, 0.0, 0.0
    length_of_collections = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    ## data load
    with open('C:/Users/helen\Desktop/NLP/tabular_data/TL_tableqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## embeddings
    emb_path = "jhgan/ko-sroberta-multitask"
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)

    ## reranking model
    reranker = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker')
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')

    # 데이터 불러오기
    for dt in data['data']:
        ids = []
        idx = 0
        questions = []
        answers = []
        context = dt['paragraphs'][0]['context']

        # 표 길이가 7이상인 것 확인
        try:
            splits = HTML_Processor.get_table_data(context)
        except:
            continue
        if len(splits) < 10 or len(splits[0]) <= 2:
            continue

        # convert 안되는 거 확인
        try:
            splits = table_to_sentence(splits)
        except:
            continue

        ## 이 테이블이 col name을 가지고 있는지, row name을 가지고 있는지
        ## 0 => row name, 1=> col name
        col_find = []
        pairs = [[splits[0][1], splits[0][2]], [splits[0][1], splits[1][1]]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=64)
            scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())
            col_find.append(np.argmax(scores))

        if col_find[0] == 0:
            continue

        reordering = LongContextReorder()

        # 1: 전처리x
        table_chunk = HTML_Processor.get_table_data(context)
        table_chunk = table_to_sentence(table_chunk)

        for _ in range(len(table_chunk)):
            ids.append(str(idx))
            idx += 1

        client1 = chromadb.PersistentClient(
            path="./patent_db_test5",
            settings=chromadb.Settings(allow_reset=True),
        )
        client1.reset()

        collection1 = client1.get_or_create_collection(
            name="table5_",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embeddings
        )
        # DB 저장
        collection1.add(
            documents=table_chunk,
            ids=ids,
        )
        length_of_collections.append(len(splits))

        # 공통 질문
        for q in dt['paragraphs'][0]['qas']:
            question = q['question']
            answer = q['answers']['text']
            questions.append(question)
            answers.append(answer)
            break

        ## 성능평가
        for i, ques in enumerate(questions):
            if ques_cnt % 10 == 0 and ques_cnt > 0:
                print('=============================')
                print(f'{ques_cnt}번째 serach 중입니다.\n'
                      f'1: 현재 top-1 정확도는 {score1_1 / ques_cnt} 입니다.'
                      f'1: 현재 top-2 정확도는 {score1_2 / ques_cnt} 입니다.'
                      f'1: 현재 top-3 정확도는 {score1_3 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
                      f'2: 현재 top-1 정확도는 {score2_1 / ques_cnt} 입니다.'
                      f'2: 현재 top-2 정확도는 {score2_2 / ques_cnt} 입니다.'
                      f'2: 현재 top-3 정확도는 {score2_3 / ques_cnt} 입니다. 평균 시간: {times2 / (ques_cnt)}\n'
                      f'3: 현재 top-1 정확도는 {score3_1 / ques_cnt} 입니다.'
                      f'3: 현재 top-2 정확도는 {score3_2 / ques_cnt} 입니다.'
                      f'3: 현재 top-3 정확도는 {score3_3 / ques_cnt} 입니다. 평균 시간: {times3 / (ques_cnt)}\n'
                      f'4: 현재 top-1 정확도는 {score4_1 / ques_cnt} 입니다.'
                      f'4: 현재 top-2 정확도는 {score4_2 / ques_cnt} 입니다.'
                      f'4: 현재 top-3 정확도는 {score4_3 / ques_cnt} 입니다. 평균 시간: {times4 / (ques_cnt)}\n'
                      )

            # 1: 바로 topk
            start_time = time.time()
            item1 = collection1.query(
                query_texts=[ques],
                n_results=3,
            )
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score1_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score1_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score1_3 += f1_score(gpt_ans3, answers[i])
            times1 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 2: 리오더 - topk
            start_time = time.time()
            item2 = collection1.query(
                query_texts=[ques],
                n_results=collection1.count(),
            )
            reordered_docs1 = reordering.transform_documents(item2['documents'])

            selected_list = [reordered_docs1[0][0], reordered_docs1[0][-1], reordered_docs1[0][1]]
            scheme = []
            for v in selected_list:
                scheme.append(v)

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score2_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score2_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score2_3 += f1_score(gpt_ans3, answers[i])
            times2 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 3: re-top-re
            start_time = time.time()
            pairs = []
            for item in selected_list:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score3_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score3_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score3_3 += f1_score(gpt_ans3, answers[i])
            times3 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

            # 4
            start_time = time.time()
            scheme = []
            for v in item1['documents'][0]:
                scheme.append(v)

            pairs = []
            for item in scheme:
                pairs.append([ques, item])

            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = exp_normalize(scores.numpy())

                indexing = np.argsort(scores)[::-1]

            scheme = [pairs[i][1] for i in indexing]

            # top-1
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:1]}"}
                ]
            )
            gpt_ans1 = conversation["choices"][0]["message"]["content"]
            score4_1 += f1_score(gpt_ans1, answers[i])
            time.sleep(random.randint(3, 6))

            # top-2
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme[:2]}"}
                ]
            )
            gpt_ans2 = conversation["choices"][0]["message"]["content"]
            score4_2 += f1_score(gpt_ans2, answers[i])
            time.sleep(random.randint(3, 6))

            # top-3
            conversation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "너는 유능한 검색기야. 다음 질문에 대해서 주어지는 내용을 참고해서 답해줘. 만약 답을 알 수 없다면 '모름'이라고 답하고, 답을 알겠다면 답인 단어만 말해줘."},
                    {"role": "user", "content": f"{ques}"},
                    {"role": "assistant", "content": f"{scheme}"}
                ]
            )
            gpt_ans3 = conversation["choices"][0]["message"]["content"]
            score4_3 += f1_score(gpt_ans3, answers[i])
            times4 += (time.time() - start_time)
            time.sleep(random.randint(3, 6))

        if ques_cnt >= 100:
            break
        ques_cnt += 1

    print(f'최종 {ques_cnt}번째 serach 중입니다.\n'
          f'1: 현재 top-1 정확도는 {score1_1 / ques_cnt} 입니다.'
          f'1: 현재 top-2 정확도는 {score1_2 / ques_cnt} 입니다.'
          f'1: 현재 top-3 정확도는 {score1_3 / ques_cnt} 입니다. 평균 시간: {times1 / (ques_cnt)}\n'
          f'2: 현재 top-1 정확도는 {score2_1 / ques_cnt} 입니다.'
          f'2: 현재 top-2 정확도는 {score2_2 / ques_cnt} 입니다.'
          f'2: 현재 top-3 정확도는 {score2_3 / ques_cnt} 입니다. 평균 시간: {times2 / (ques_cnt)}\n'
          f'3: 현재 top-1 정확도는 {score3_1 / ques_cnt} 입니다.'
          f'3: 현재 top-2 정확도는 {score3_2 / ques_cnt} 입니다.'
          f'3: 현재 top-3 정확도는 {score3_3 / ques_cnt} 입니다. 평균 시간: {times3 / (ques_cnt)}\n'
          f'4: 현재 top-1 정확도는 {score4_1 / ques_cnt} 입니다.'
          f'4: 현재 top-2 정확도는 {score4_2 / ques_cnt} 입니다.'
          f'4: 현재 top-3 정확도는 {score4_3 / ques_cnt} 입니다. 평균 시간: {times4 / (ques_cnt)}\n'
          )
    plt.hist(length_of_collections, bins=len(length_of_collections)-10, color='green', edgecolor='black')

    # 그래프 제목과 축 라벨 추가
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # 그래프 출력
    plt.show()

# no_pre_model()
# table_row_model()
# test_no_pre_model()
# test_table_chunk_model()
test_table_row_model()