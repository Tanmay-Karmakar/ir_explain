import numpy as np

def get_results_from_index(query, index_searcher, num_docs=10):
    #Perform sparse retrieval to get 'num_docs' number of documents from the index specified by 'index_searcher'
    
    hits = index_searcher.search(query, num_docs)
    docs = []
    doc_ids = []
    retrieved_scores = []
    for hit in hits:
        #doc_text = (hit.docid + " " + hit.raw.replace("\n"," ")).strip()
        #print(hit.contents)
        #doc_text = hit.contents() # .replace("\n"," ").strip()
        doc_text = hit.lucene_document.get('raw').replace("\n"," ").strip()
        docs.append(doc_text)
        doc_ids.append(hit.docid)
        retrieved_scores.append(hit.score)
    print(docs)
    print(doc_ids)

    retrieved_dict = {'doc_ids' : np.array(doc_ids), 'docs' : np.array(docs), 'retrieved_scores' : np.array(retrieved_scores)}

    return retrieved_dict