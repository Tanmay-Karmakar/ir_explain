import json
import numpy as np
from intent_exs import IntentEXS
from pyserini.search.lucene import LuceneSearcher

index_path = "/b/administrator/collections/indexed/msmarco-v1-passage-full"
searcher = LuceneSearcher(index_path)   # load a searcher from pre-computed index.

res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/tct-colbert-hnsw-trecdl-2020.res"

from utilities import utility

dense_ranking, dense_scores = utility.load_from_res(res_file_path)

query_str = "causes of stroke?"
qid = "88495"


dense_ranking_list = dense_ranking['88495']
dense_score_list = dense_scores['88495']

docs = dict([(hit, json.loads(searcher.doc(hit).raw())['contents']) for hit in dense_ranking_list[:20]])

# Load a reranking model
from beir.reranking.models import CrossEncoder
model = 'cross-encoder/ms-marco-electra-base'
reranker = CrossEncoder(model)

corpus = {'query': query_str,
        'scores': dict([(doc_id, score) for doc_id, score in zip(dense_ranking_list[:20], dense_score_list[:20])]),
        'docs': docs
    }

params = {'top_idf': 200, 'topk': 20, 'max_pair': 100, 'max_intent': 20, 'style': 'random'}


# Init the IntentEXS object.
Intent = IntentEXS(reranker, index_path, 'bm25')

expansion = Intent.explain(corpus, params)




