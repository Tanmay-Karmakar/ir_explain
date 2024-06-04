params = {
    "dataset" : "clueweb09",
    "top_d" : 10,
    "top_tfidf" : 100,
    "top_r" : 50,
    "candi_method" : "bm25",
    "ranked" : 5,
    "pair_num" : 20,
    "max_k" : 10,
    "min_k" : 4,
    "candidates_num" : 20,
    "style" : "random",
    "vote" : 1,
    "tolerance" : .005,
    "EXP_model" : "language_model",
    "optimize_method" : "geno",
    "mode" : "candidates"
    }

params ["top_file"] = "/a/administrator/codebase/neural-ir/RankingExplanation_bkp/Datasets/src/clueweb09/top.tsv"
params ["queries_file"] = "/a/administrator/codebase/neural-ir/RankingExplanation_bkp/Datasets/src/clueweb09/queries.tsv"


query_str = 'lps laws definition'
qid = '443396'



query_str = "causes of stroke?"
qid = "88495"

query_str = "what is the daily life of thai people"
qid = '1112341'


query_str = "what causes heavy metal toxins in your body"
qid = "588587"



res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/ANCE.2019.res"
res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/tct-colbert-hnsw-trecdl-2020.res"

index_path = "/b/administrator/collections/indexed/msmarco-v1-passage-full"

def load_from_res(res_file_path):
    qid_docid_list = {}
    qid_docscore_list = {}

    res_file = open(res_file_path, 'r')
    lines = res_file.readlines()

    for line in lines:
        qid, _, docid, rank, score, _ = line.split()
        list_of_docids = []
        list_of_docscores = []
        if qid in qid_docid_list:
            list_of_docids = qid_docid_list[qid]
            list_of_docscores = qid_docscore_list[qid]

        list_of_docids.append(docid)
        list_of_docscores.append(score)

        qid_docid_list[qid] = list_of_docids
        qid_docscore_list[qid] = list_of_docscores

    return qid_docid_list, qid_docscore_list



dense_ranking, dense_scores = load_from_res(res_file_path)

dense_ranking_list = dense_ranking['443396']
dense_score_list = dense_scores['443396']


dense_ranking_list = dense_ranking['88495']
dense_score_list = dense_scores['88495']


dense_ranking_list = dense_ranking['1112341']
dense_score_list = dense_scores['1112341']


dense_ranking_list = dense_ranking['588587']
dense_score_list = dense_scores['588587']






params["dense_ranking"] = dense_ranking_list

params["dense_ranking_score"] = dense_score_list



from explain_run_custom import Multiplex

multi = Multiplex(index_path)
params["EXP_model"] = "multi"
params["optimize_method"] = "geno_multi"

multi.explain(qid, query_str, params)



query_str = 'obama family tree'
qid = '1'




from bfs_explainer import BFS

params = {
    "QUEUE_MAX_DEPTH" : 1000,
    "BFS_MAX_EXPLORATION" : 30,
    "BFS_VOCAB_TERMS" : 30,
    "BFS_MAX_DEPTH" : 10,
    "BFS_TOP_DOCS" : 10,
    }

bfs = BFS(index_path, exp_model, params)


def load_from_res(res_file_path):
    qid_docid_list = {}
    
    res_file = open(res_file_path, 'r')
    lines = res_file.readlines() 
    
    for line in lines:
        qid, _, docid, rank, score, _ = line.split()
        list_of_docids = []
        if qid in qid_docid_list:
            list_of_docids = qid_docid_list[qid]

        list_of_docids.append(docid)
        qid_docid_list[qid] = list_of_docids
    
    return qid_docid_list

dense_ranking = load_from_res(res_file_path)



query_id = 
query_str 

searcher = LuceneSearcher("/b/administrator/collections/indexed/msmarco-v1-passage-full")

searcher.set_bm25(1.2, 0.75)     # set BM25 parameter
searcher.set_analyzer(get_lucene_analyzer(stemmer='porter'))
bm25_hits = searcher.search(query_str)


searcher.set_rm3(1000, 10, 0.9)   # set parameter for rm3



term_weight_list = searcher.get_feedback_terms(query_str)
term_weight_list = dict(sorted(term_weight_list.items(), key=lambda item: item[1], reverse = True))




bfs.explain(query_id, query_str, term_weight_list, searcher, dense_ranking)


from greedy_explainer import Greedy

params = {
    "GREEDY_VOCAB_TERMS" : 100,
    "GREEDY_TOP_DOCS_NUM" : 10,
    "GREEDY_MAX_DEPTH" : 10,
    "BFS_TOP_DOCS" : 10,
    }

greedy = Greedy(index_path, exp_model, params)

greedy.explain(query_id, query_str, term_weight_list, searcher, dense_ranking)



import json
import numpy as np
from intent_exs import IntentEXS
from pyserini.search.lucene import LuceneSearcher  

index_path = "/b/administrator/collections/indexed/msmarco-v1-passage-full"
searcher = LuceneSearcher(index_path)   # load a searcher from pre-computed index.

res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/tct-colbert-hnsw-trecdl-2020.res"

def load_from_res(res_file_path):
    qid_docid_list = {}
    qid_docscore_list = {}

    res_file = open(res_file_path, 'r')
    lines = res_file.readlines()

    for line in lines:
        qid, _, docid, rank, score, _ = line.split()
        list_of_docids = []
        list_of_docscores = []
        if qid in qid_docid_list:
            list_of_docids = qid_docid_list[qid]
            list_of_docscores = qid_docscore_list[qid]

        list_of_docids.append(docid)
        list_of_docscores.append(score)

        qid_docid_list[qid] = list_of_docids
        qid_docscore_list[qid] = list_of_docscores

    return qid_docid_list, qid_docscore_list



dense_ranking, dense_scores = load_from_res(res_file_path)
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

