import json
import numpy as np
from intent_exs import IntentEXS
from pyserini.search.lucene import LuceneSearcher

# MSMARCO index path: 
# one can download it from pyserini
index_path = "/path/to/index/stored/msmarco-v1-passage-full"
searcher = LuceneSearcher(index_path)   # load a searcher from pre-computed index.

# provide res file path of the blackbox ranker
res_file_path = "/path_to_ir_explain_folder/ir_explain/runs/NRMs/tct-colbert-hnsw-trecdl-2020.res"

from utilities import utility

# load dense ranking result and scores
dense_ranking, dense_scores = utility.load_from_res(res_file_path)

# provide query for which listwise explainer BFS needs to be invoked
# one can fetch query and query id from ir_datasets (https://ir-datasets.com/)
query_str = "causes of stroke?"
qid = "88495"



dense_ranking_list = dense_ranking['88495']
dense_score_list = dense_scores['88495']

# for the top k documents fetch their contents
docs = dict([(hit, json.loads(searcher.doc(hit).raw())['contents']) for hit in dense_ranking_list[:20]])

# Load a reranking model
from beir.reranking.models import CrossEncoder
model = 'cross-encoder/ms-marco-electra-base'
reranker = CrossEncoder(model)

corpus = {'query': query_str,
        'scores': dict([(doc_id, score) for doc_id, score in zip(dense_ranking_list[:20], dense_score_list[:20])]),
        'docs': docs
    }

# set parameters fro IntentEXS
params = {'top_idf': 200, 'topk': 20, 'max_pair': 100, 'max_intent': 20, 'style': 'random'}


# Init the IntentEXS object.
Intent = IntentEXS(reranker, index_path, 'bm25')

# call explain method of IntentEXS
expansion = Intent.explain(corpus, params)




