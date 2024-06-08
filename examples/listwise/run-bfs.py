from utilities import utility
from bfs_explainer import BFS
from pyserini.search.lucene import LuceneSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer

# provide query for which listwise explainer BFS needs to be invoked
# one can fetch query and query id from ir_datasets (https://ir-datasets.com/) 
query_str = "what is the daily life of thai people"

qid = '1112341'
query_id = '1112341'

# provide res file path of the blackbox ranker
res_file_path = "/path_to_ir_explain_folder/ir_explain/runs/NRMs/ANCE.2019.res"

# MSMARCO index path: 
# one can download a pre-built index from pyserini
index_path = "/path/to/index/stored/msmarco-v1-passage-full"

# load dense ranking result and scores
dense_ranking, dense_scores = utility.load_from_res(res_file_path)

dense_ranking_list = dense_ranking['1112341']
dense_score_list = dense_scores['1112341']

# initialize the parameters of BFS
params = {
    "QUEUE_MAX_DEPTH" : 1000,
    "BFS_MAX_EXPLORATION" : 30,
    "BFS_VOCAB_TERMS" : 30,
    "BFS_MAX_DEPTH" : 10,
    "BFS_TOP_DOCS" : 10,
    "CORRELATION_MEASURE" : "RBO",
    }

exp_model = "bm25"

# initialize the BFS class
bfs = BFS(index_path, exp_model, params)

# initialize LuceneSearcher
# we use LuceneSearcher from pyserini
searcher = LuceneSearcher(index_path)


searcher.set_bm25(1.2, 0.75)     # set BM25 parameter
searcher.set_analyzer(get_lucene_analyzer(stemmer='porter'))
# retrieve with BM25 
bm25_hits = searcher.search(query_str)

# set parameters for RM3
searcher.set_rm3(1000, 10, 0.9)

# generate the feedback terms 
term_weight_list = searcher.get_feedback_terms(query_str)

# sort the feedback terms
term_weight_list = dict(sorted(term_weight_list.items(), key=lambda item: item[1], reverse = True))

# call BFS explainer module
bfs.explain(query_id, query_str, term_weight_list, searcher, dense_ranking, debug = False)


