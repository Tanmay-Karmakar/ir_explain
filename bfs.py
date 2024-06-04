import ir_measures, ir_datasets
from ir_measures import *
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
from queue import PriorityQueue
import utilities.rbo as rbo
import random, sys
from pyserini.index.lucene import IndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer

# traditional model: yes
# LSR : 
# DR : 
# hybrid models : 

# change dataset name accordingly 
dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
# NOTE: there is a confusion b/w msmarco passage collection v1 and v2
searcher = LuceneSearcher("/b/administrator/collections/indexed/msmarco-v1-passage-full")
index_reader = IndexReader("/b/administrator/collections/indexed/msmarco-v1-passage-full")
# TODO: set the analyzer properly
#searcher.set_analyzer()

# hard-coded as of now: TODO : change it later
qrel_path = "/b/administrator/collections/qrels/2019qrels-pass.txt"
query_path = "/b/administrator/collections/queries/msmarco-test2019-queries.tsv"
output_path = "/tmp/bfs_"
#res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/BM25_deepct_ColBERT.2019.res"
#res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/colberte2e_bertqe_sorted.2019.res"
#res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/ANCE.2019.res"
res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/ColbertE2E_sorted.2019.res"
#res_file_path = "/a/administrator/codebase/neural-ir/ir_explain/runs/NRMs/trecdl.monot5.rr.pos-scores.res"



searcher.set_bm25(1.2, 0.75)     # set BM25 parameter
#searcher.set_bm25()     # set BM25 parameter
searcher.set_analyzer(get_lucene_analyzer(stemmer='porter'))

# TODO : check the param value should be 0.9 or .01

# retrieve with a complex ranker

#encoder = AnceQueryEncoder('castorini/ance-msmarco-passage')
#dr_searcher = FaissSearcher.from_prebuilt_index(
#    'ance-msmarco-passage',
#    encoder
#)

# retrieve with a complex ranker; taking too much memory TODO : initialize it later 
#encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
#dr_searcher = FaissSearcher(
#    '/b/administrator/collections/indexed/msmarco-passage-tct_colbert-hnsw/',
#    encoder
#)


#dr_hits = dr_searcher.search('what is a lobster roll')

# constants
QUEUE_MAX_DEPTH = 1000 # 10
BFS_MAX_EXPLORATION = 25
BFS_VOCAB_TERMS = 30 # 30
BFS_MAX_DEPTH = 10
BFS_TOP_DOCS = 10


class DualPriorityQueue(PriorityQueue):
    def __init__(self, maxSize = 10, maxPQ=False):
        PriorityQueue.__init__(self, maxsize=maxSize)
        self.reverse = -1 if maxPQ else 1

    def put(self, data):
        PriorityQueue.put(self, (self.reverse * data[0], data[1]))

    def get(self, *args, **kwargs):
        priority, data = PriorityQueue.get(self, *args, **kwargs)
        return self.reverse * priority, data



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

    #print(len(qid_docid_list.keys()))
    #for key in qid_docid_list.keys():
    #    print(len(qid_docid_list[key]))

    #print(len(qid_docid_list))
    #sys.exit(1)
    return qid_docid_list

# TODO: pass flag: which similarity measure to use:
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

def sample_terms(term_weight_list, num_terms):
    """
    Sample terms from the feedback terms
    term_weight_list : sorted by weights
    """
    total_weight = 0                                # ideally total_weight should be 1
    for value in term_weight_list.values():
        total_weight += value
    terms_list = []

    print(f'Sampling terms...')
    while len(terms_list) < num_terms and len(terms_list) < len(term_weight_list):
        index = random.uniform(0, 1)*total_weight

        for term, weight in term_weight_list.items():
            index = index - weight

            if index <= 0 :
                if term not in terms_list:
                    terms_list.append(term)
                    break
    
    print('Sampling completed...')
    return terms_list
    

def bfs(qid, query_str, term_weight_list, searcher, dense_ranking):
    """
    bfs algorithm to generate the expanded query
    Paper link: https://arxiv.org/pdf/2304.12631.pdf
    """
    state = ""
    expanded_query = ""
    best_state = None
    searcher.unset_rm3()
    # for debug purpose:
    print('Using Rm3? ', searcher.is_using_rm3())  # this should be false at this moment
    print(f'Entire query string {query_str}')    
    # analyzed_terms = index_reader.analyze(query_str)

    for term in query_str.split():
    #for term in analyzed_terms:
        print(f'running for query term {term}')
        print(f'original qid {qid}\nquery {query_str}')
        #print(query_str.split())
        maxQ = DualPriorityQueue(QUEUE_MAX_DEPTH, maxPQ=True)
        bm25_hits = searcher.search(term)
        print(len(bm25_hits))
        if len(bm25_hits) == 0:
            continue
        similarity_rbo = compute_rbo(bm25_hits, dense_ranking)
        # initial_state = tuple((term, similarity_rbo))                   # hack : put priority as negative
        initial_state = tuple((similarity_rbo, term))                   

        if best_state is None:
            best_state = initial_state
        
        maxQ.put(initial_state)
        states_explored = 0

        print(f'queue at start : {dict(maxQ.queue)}')
        while (not maxQ.empty()) and (states_explored < BFS_MAX_EXPLORATION):
            current_best = maxQ.get()
            states_explored += 1
            
            if current_best[0] > best_state[0]:
                best_state = current_best
            
            # sample 30 terms from the feedbackDocs
            sampled_terms = sample_terms(term_weight_list, BFS_VOCAB_TERMS)
            print(f'size of sampled terms {len(sampled_terms)}')
            #print(sampled_terms)
            
            for vocab_term in sampled_terms:
                new_query = ""

                #print(f"term want to add {vocab_term}")
                if vocab_term not in current_best[1]:
                    new_query = current_best[1] + " " + vocab_term
                else:
                    new_query = current_best[1]
                
                print(f"new query : {new_query}")
                print(f'queue : {dict(maxQ.queue)}')
                if  new_query not in dict(maxQ.queue) and len(new_query.split()) < BFS_MAX_DEPTH:
                    # retrieve with the new expanded query
                    new_top_docs = searcher.search(new_query, BFS_TOP_DOCS)
                    new_similarity_rbo = compute_rbo(new_top_docs, dense_ranking)
        
                    if new_similarity_rbo >= current_best[0] and new_similarity_rbo > 0:
                        element = tuple((new_similarity_rbo, new_query))
                        #print(f"adding {element} to the queue")
                        
                        #print(f"Size of queue {len(dict(q.queue))} earlier")
                        #print(f"States explored {states_explored}")
                        maxQ.put(element)
                        #print(f"Size of queue {len(dict(q.queue))} later")
                        #print(f'queue : {dict(q.queue)}')

                    if new_similarity_rbo > best_state[0]:
                        best_state = tuple((new_similarity_rbo, new_query))
                        print(f'best state as of now {best_state}')

        print(f'max exploration done {states_explored}')
        #print(f'size of queue {q.qsize}')
    return best_state

count = 0
results = [] 
avg_rbo = 0
for query in dataset.queries_iter():
    count += 1
    query_str = query.text
    qid = query.query_id
    print(f'Running for query id : {qid}\n{query_str}')
    
    bm25_hits = searcher.search(query_str)
    dense_ranking = load_from_res(res_file_path)
    
    #dr_hits = dr_searcher.search(query_str)
    #dense_ranking_list = []
    #for index in range(len(dr_hits)):
    #    dense_ranking_list.append(dr_hits[index].docid)

    #dr_hits = searcher.search(query_str)
    searcher.set_rm3(200, 10, 0.7)   # set parameter for rm3

    term_weight_list = searcher.get_feedback_terms(query_str)
    print(term_weight_list)
    term_weight_list = dict(sorted(term_weight_list.items(), key=lambda item: item[1]))
    best_state = bfs(qid, query_str, term_weight_list, searcher, dense_ranking[qid])
    #best_state = tuple((query_str, 0))              # just for debug: TODO uncomment it

    # print qid, expanded query, RBO, map 
    print(best_state[1], "\t", best_state[0], "\n")
    
    # search with the expanded query formed by bfs
    final_hits = searcher.search(best_state[1])
    #final_hits = dr_searcher.search(best_state[0])          # just for sanity check 
    avg_rbo += float(best_state[0])
    results.append(tuple((qid, final_hits)))

    #break

print(f'Total number of query {count}')
avg_rbo = avg_rbo/count
print(f'RBO for the entire query set {avg_rbo}')

#print(f'length of the result file : {len(results)}')
qrels = ir_measures.read_trec_qrels(qrel_path)
tag = "bfs-qe"
output_path = output_path + res_file_path.split("/")[-1] + ".reproduced"
f = open(output_path, "w")
for qid, hits in results:
    for i in range(0, len(hits)):
        f.write(f'{qid} Q0 {hits[i].docid} {i} {hits[i].score} {tag}\n')

f.close()

run = ir_measures.read_trec_run(output_path)   
output_result = ir_measures.calc_aggregate([nDCG@10, AP], qrels, run)
print(output_result)
