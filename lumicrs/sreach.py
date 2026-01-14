from rank_bm25 import BM25Okapi
from transformers import DPRContextEncoder, DPRQuestionEncoder
import numpy as np

# 加载数据
valid_texts = [item['text'] for item in valid_data]
test_texts = [item['text'] for item in test_data]

# Step 1: BM25 初步检索
bm25 = BM25Okapi([text.split() for text in valid_texts])


# BM25 检索函数
def bm25_retrieval(query, corpus):
    return bm25.get_top_n(query.split(), corpus, n=10)


# Step 2: DPR 精细化检索
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")


def dpr_retrieval(query, corpus, context_encoder, question_encoder):
    query_embedding = question_encoder.encode([query])[0]
    context_embeddings = [context_encoder.encode([doc])[0] for doc in corpus]
    similarities = np.dot(context_embeddings, query_embedding.T)
    return sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:10]


# Step 3: 混合检索（BM25 + DPR）
def hybrid_task_driven_retrieval(query, corpus, use_dpr=True):
    bm25_results = bm25_retrieval(query, corpus)

    if use_dpr:
        dpr_results = dpr_retrieval(query, corpus, context_encoder, question_encoder)
        combined_results = bm25_results + [corpus[i[0]] for i in dpr_results]
    else:
        combined_results = bm25_results

    return combined_results


# 示例查询
query = test_texts[0]  # 示例查询
results = hybrid_task_driven_retrieval(query, valid_texts)

# 输出检索结果
print("Retrieved Results:", results)
