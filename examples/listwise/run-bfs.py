from utilities import utility
from bfs_explainer import BFS
from pyserini.search.lucene import LuceneSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer


query_str = "what is the daily life of thai people"

qid = '1112341'
query_id = '1112341'

res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/ANCE.2019.res"

index_path = "/b/administrator/collections/indexed/msmarco-v1-passage-full"

dense_ranking, dense_scores = utility.load_from_res(res_file_path)

dense_ranking_list = dense_ranking['1112341']
dense_score_list = dense_scores['1112341']

params = {
    "QUEUE_MAX_DEPTH" : 1000,
    "BFS_MAX_EXPLORATION" : 30,
    "BFS_VOCAB_TERMS" : 30,
    "BFS_MAX_DEPTH" : 10,
    "BFS_TOP_DOCS" : 10,
    "CORRELATION_MEASURE" : "RBO",
    }

exp_model = "bm25"

bfs = BFS(index_path, exp_model, params)

searcher = LuceneSearcher("/b/administrator/collections/indexed/msmarco-v1-passage-full")

searcher.set_bm25(1.2, 0.75)     # set BM25 parameter
searcher.set_analyzer(get_lucene_analyzer(stemmer='porter'))
bm25_hits = searcher.search(query_str)


searcher.set_rm3(1000, 10, 0.9)   # set parameter for rm3

term_weight_list = searcher.get_feedback_terms(query_str)
term_weight_list = dict(sorted(term_weight_list.items(), key=lambda item: item[1], reverse = True))

bfs.explain(query_id, query_str, term_weight_list, searcher, dense_ranking, debug = False)


