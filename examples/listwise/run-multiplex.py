# set parameters for Multiplex
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

# pass the topk file
params ["top_file"] = "/a/administrator/codebase/neural-ir/RankingExplanation_bkp/Datasets/src/clueweb09/top.tsv"
# pass the query file
params ["queries_file"] = "/a/administrator/codebase/neural-ir/RankingExplanation_bkp/Datasets/src/clueweb09/queries.tsv"

# provide query for which listwise explainer BFS needs to be invoked
# one can fetch query and query id from ir_datasets (https://ir-datasets.com/)
query_str = 'lps laws definition'
qid = '443396'

# provide res file path of the blackbox ranker
res_file_path = "/path_to_ir_explain_folder/ir_explain/runs/NRMs/ANCE.2019.res"

# MSMARCO index path:
# one can download it from pyserini
index_path = "/path/to/index/stored/msmarco-v1-passage-full"

from utilities import utility

# load dense ranking result and scores
dense_ranking, dense_scores = utility.load_from_res(res_file_path)

dense_ranking_list = dense_ranking['443396']
dense_score_list = dense_scores['443396']

params["dense_ranking"] = dense_ranking_list
params["dense_ranking_score"] = dense_score_list


from explain_run_custom import Multiplex

# initialize the Multiplex class
multi = Multiplex(index_path)

params["EXP_model"] = "multi"
params["optimize_method"] = "geno_multi"


# call Multiplex explainer module
multi.explain(qid, query_str, params)

# DEBUG/EXPLAIN Multiplex with additional details
# e.g., 1) generate_candidates 2) generate_doc_pairs, 3) show_matrix
multi.generate_candidates(qid, query_str, params)

multi.generate_doc_pairs(qid, query_str, params)

multi.show_matrix(qid, query_str, params)


