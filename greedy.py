import ir_measures, ir_datasets
from ir_measures import *
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
from queue import PriorityQueue
import utilities.rbo as rbo
from pyserini.index.lucene import IndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer
import heapq

# change dataset name accordingly 
dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
# NOTE: there is a confusion b/w msmarco passage collection v1 and v2
searcher = LuceneSearcher("/b/administrator/collections/indexed/msmarco-v1-passage-full")
index_reader = IndexReader("/b/administrator/collections/indexed/msmarco-v1-passage-full")
# TODO: set the analyzer properly
# hard-coded as of now: TODO : change it later
qrel_path = "/b/administrator/collections/qrels/2019qrels-pass.txt"
query_path = "/b/administrator/collections/queries/msmarco-test2019-queries.tsv"
output_path = "/tmp/greedy.res.foo"
res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/colberte2e_bertqe_sorted.2019.res"
#res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/ANCE.2019.res"


searcher.set_bm25(1.2, 0.75)     # set BM25 parameter
#searcher.set_bm25()     # set BM25 parameter
searcher.set_analyzer(get_lucene_analyzer(stemmer='porter'))

#encoder = AnceQueryEncoder('castorini/ance-msmarco-passage')
#dr_searcher = FaissSearcher.from_prebuilt_index(
#    'ance-msmarco-passage',
#    encoder
#)

#dr_hits = dr_searcher.search('what is a lobster roll')

# constants
GREEDY_VOCAB_TERMS = 100
BFS_TOP_DOCS = GREEDY_TOP_DOCS_NUM = 10
GREEDY_MAX_DEPTH = 10


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


def compute_rbo(bm25_hits, dr_hits):
    """
    Compute rbo between two ranked list 
    """
    p = 0.9
    depth = 10
    
    bm25_list = []
    dr_list = dr_hits[:BFS_TOP_DOCS]

    for i in range(0, min(BFS_TOP_DOCS, len(bm25_hits))):
        bm25_list.append(bm25_hits[i].docid)
        #dr_list.append(dr_hits[i].docid)
        #dr_list.append(dr_hits[i])

    #print(bm25_list)
    #print(dr_list)

    rbo_score = rbo.RankingSimilarity(bm25_list, dr_list).rbo(p = 0.9)
    print(f'rbo score {rbo_score}')
    return rbo_score

def greedy(qid, query_str, term_weight_list, searcher, dense_ranking):
    """
    bfs algorithm to generate the expanded query
    Paper link: https://arxiv.org/pdf/2304.12631.pdf
    """
    searcher.unset_rm3()
    # for debug purpose:
    print('Using Rm3? ', searcher.is_using_rm3())  # this should be false at this moment
    print(f'Entire query string {query_str}')    

    #term_weight_list_till_k = heapq.nlargest(min(len(term_weight_list), GREEDY_VOCAB_TERMS), term_weight_list)
    topk_terms = list()
    
    top_k = min(len(term_weight_list), GREEDY_VOCAB_TERMS)

    term_weight_list_till_k = dict(sorted(term_weight_list.items(), key=lambda item: item[1], reverse = True)[:top_k])

    for term in query_str.split():
        print(f'running for query term {term}')

        bm25_hits = searcher.search(term)
        print(len(bm25_hits))
        if len(bm25_hits) == 0:
            continue
        similarity_rbo = compute_rbo(bm25_hits, dense_ranking[qid])
        
        for key in term_weight_list_till_k:
            new_query = term + " " + key
            new_hits = searcher.search(new_query, GREEDY_TOP_DOCS_NUM)
            new_similarity_rbo = compute_rbo(new_hits, dense_ranking[qid])

            topk_terms.append(tuple((key, new_similarity_rbo - similarity_rbo)))

    topk_terms.sort(key = lambda tup: tup[1], reverse = True)
    final_query = ""

    i = 0
    while len(final_query.split()) < GREEDY_MAX_DEPTH:  
        print(f'length of topk_terms vector {len(topk_terms)}')
        print(topk_terms)
        term, rbo_contribution = topk_terms[i]

        if term not in final_query:
            final_query = final_query + " " + term
        i = i + 1 
    
    final_hits = searcher.search(final_query)
    final_similarity = compute_rbo(final_hits, dense_ranking[qid])

    return tuple((final_similarity, final_query))

count = 0
results = [] 
for query in dataset.queries_iter():
    count += 1
    query_str = query.text
    qid = query.query_id
    print(f'Running for query id : {qid}\n{query_str}')
    
    bm25_hits = searcher.search(query_str)
    dense_ranking = load_from_res(res_file_path)
    
    #dr_hits = dr_searcher.search(query_str)
    #dr_hits = searcher.search(query_str)
    searcher.set_rm3(1000, 10, 0.9)   # set parameter for rm3

    term_weight_list = searcher.get_feedback_terms(query_str)
    print(term_weight_list)
    
    best_state = greedy(qid, query_str, term_weight_list, searcher, dense_ranking)
    #best_state = tuple((query_str, 0))              # just for debug: TODO uncomment it

    # print qid, expanded query, RBO, map 
    print(best_state[1], "\t", best_state[0], "\n")
    
    # search with the expanded query formed by bfs
    final_hits = searcher.search(best_state[1])
    #final_hits = dr_searcher.search(best_state[0])          # just for sanity check 

    results.append(tuple((qid, final_hits)))
    
    #break

print(f'Total number of query {count}')

#print(f'length of the result file : {len(results)}')
qrels = ir_measures.read_trec_qrels(qrel_path)
tag = "greedy-qe"
f = open(output_path, "w")
for qid, hits in results:
    for i in range(0, len(hits)):
        f.write(f'{qid} Q0 {hits[i].docid} {i} {hits[i].score} {tag}\n')

f.close()

run = ir_measures.read_trec_run(output_path)   
output_result = ir_measures.calc_aggregate([nDCG@10, AP], qrels, run)
print(output_result)