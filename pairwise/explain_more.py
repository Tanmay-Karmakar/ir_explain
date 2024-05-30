import itertools
import pandas as pd

def calculate_avg_distance(occurrences):
    distances = [occurrences[i+1] - occurrences[i] for i in range(len(occurrences)-1)]
    return sum(distances) / len(distances) if distances else 0

class explain_more:

    class TFC1:
    
        def explain(self,query, document1, document2):
         
            query_words = set(query.split())
            document1_words = set(document1.split())
            document2_words = set(document2.split())

            common_words1 = len(query_words.intersection(document1_words)) / len(document1_words)
            common_words2 = len(query_words.intersection(document2_words)) / len(document2_words)
            
            print("Normalised tf score in doc1: " +str(common_words1))
            print("Normalised tf score in doc2: " +str(common_words2))

    class PROX1:
    
        def explain(query, document1, document2):
            
            query_words = query.split()

           
            words_doc1 = document1.split()
            words_doc1 = [word.replace('.', '') for word in words_doc1]
            words_doc2 = document2.split()
            words_doc2 = [word.replace('.', '') for word in words_doc2]

            term_pairs = list(itertools.combinations(query_words, 2))

            avg_distances = {pair: [] for pair in term_pairs}

            for term1, term2 in term_pairs:
                
                occurrences_doc1_term1 = [i for i, w in enumerate(words_doc1) if w == term1]
                occurrences_doc1_term2 = [i for i, w in enumerate(words_doc1) if w == term2]

                occurrences_doc2_term1 = [i for i, w in enumerate(words_doc2) if w == term1]
                occurrences_doc2_term2 = [i for i, w in enumerate(words_doc2) if w == term2]

                avg_distance_doc1 = calculate_avg_distance(occurrences_doc1_term1) + calculate_avg_distance(occurrences_doc1_term2)

                avg_distance_doc2 = calculate_avg_distance(occurrences_doc2_term1) + calculate_avg_distance(occurrences_doc2_term2)

                avg_distances[(term1, term2)].extend([avg_distance_doc1, avg_distance_doc2])

            df = pd.DataFrame(avg_distances, index=['Document 1', 'Document 2']).T
            df.index = pd.MultiIndex.from_tuples(df.index, names=['Term 1', 'Term 2'])
            df.reset_index(inplace=True)
            
            return df
            
    class PROX2:
        
        def explain(query, document1, document2):
        
            query_words = query.split()

            words_doc1 = document1.split()
            words_doc1 = [word.replace('.', '') for word in words_doc1]
            words_doc2 = document2.split()
            words_doc2 = [word.replace('.', '') for word in words_doc2]

            first_occurrences = {}

            for term in query_words:
                index = next((i for i, word in enumerate(words_doc1) if word == term), None)
                first_occurrences[f'Document 1 - {term}'] = index

            for term in query_words:
                index = next((i for i, word in enumerate(words_doc2) if word == term), None)
                first_occurrences[f'Document 2 - {term}'] = index

            for key, value in first_occurrences.items():
                print(f"{key}: {value}")

    class PROX3:
    
        def explain(query, document1, document2):

            index_doc1 = document1.find(query)
            index_doc2 = document2.find(query)

            print(f"First occurrence of the query in Document 1: {index_doc1 if index_doc1 != -1 else 'Not present'}")
            print(f"First occurrence of the query in Document 2: {index_doc2 if index_doc2 != -1 else 'Not present'}")

    class LNC1:

        def explain(query, document1, document2): 

            query_words = query.split()

            count_query_terms_doc1 = sum(1 for word in query_words if word in document1)
            count_query_terms_doc2 = sum(1 for word in query_words if word in document2)

            print(f"Number of query terms document 1: {count_query_terms_doc1}")
            print(f"Number of query terms document 2: {count_query_terms_doc2}")
            print()
            print(f"Length of document1: {len(document1)}")
            print(f"Length of document2: {len(document2)}")

    class TF_LNC:

        def explain(query, document1, document2):

            query_words = set(query.split())
            document1_words = set(document1.split())
            document2_words = set(document2.split())
        
            common_words1 = query_words.intersection(document1_words)
            common_words2 = query_words.intersection(document2_words)
            
        
            words1 = document1.split()
            words2 = document2.split()
        
            filtered_words1 = [word for word in words1 if word not in common_words1]
            filtered_words2 = [word for word in words2 if word not in common_words2]
        
            new_doc1 = ' '.join(filtered_words1)
            new_doc2 = ' '.join(filtered_words2)
        
            max_len = max(len(new_doc1), len(new_doc2))  
            tolerance = 0.1 * max_len
        
            if abs(len(new_doc1) - len(new_doc2)) > tolerance:
                print("Documents are not of approximately equal length")
            else:
                print(f"Query terms in document1: {len(common_words1)}")
                print(f"Query terms in document2: {len(common_words2)}")

    class LB1:

        def explain(query, document1, document2):
    
            query_terms = set(query.lower().split())
            doc1_terms = set(document1.lower().split())
            doc2_terms = set(document2.lower().split())

            unique_to_doc1 = [term for term in query_terms if term in doc1_terms and term not in doc2_terms]
            unique_to_doc2 = [term for term in query_terms if term in doc2_terms and term not in doc1_terms]

            print(f"Query terms present in document 1 but not in document 2 {unique_to_doc1}")
            print(f"Query terms present in document 2 but not in document 1 {unique_to_doc2}")


    
           
        
