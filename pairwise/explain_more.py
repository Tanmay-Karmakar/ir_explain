import itertools
import pandas as pd

def calculate_avg_distance(occurrences):
    distances = [occurrences[i+1] - occurrences[i] for i in range(len(occurrences)-1)]
    return sum(distances) / len(distances) if distances else 0

class explain_more:

    class TFC1:
    
        def explain(self,query, document1, document2):
            # Convert the documents and query into sets of words
            query_words = set(query.split())
            document1_words = set(document1.split())
            document2_words = set(document2.split())

            # Calculate the number of common words between the query and each document
            common_words1 = len(query_words.intersection(document1_words)) / len(document1_words)
            common_words2 = len(query_words.intersection(document2_words)) / len(document2_words)
            
            print("Normalised tf score in doc1: " +str(common_words1))
            print("Normalised tf score in doc2: " +str(common_words2))

    class PROX1:
    
        def explain(query, document1, document2):
            # Tokenize the query sentence into words
            query_words = query.split()

            # Tokenize the documents into words
            words_doc1 = document1.split()
            words_doc1 = [word.replace('.', '') for word in words_doc1]
            words_doc2 = document2.split()
            words_doc2 = [word.replace('.', '') for word in words_doc2]

            # Define the pairs of terms
            term_pairs = list(itertools.combinations(query_words, 2))

            # Initialize a dictionary to store the average distances between term pairs
            avg_distances = {pair: [] for pair in term_pairs}

            # Iterate through the term pairs
            for term1, term2 in term_pairs:
                # Find occurrences of each term in document 1
                occurrences_doc1_term1 = [i for i, w in enumerate(words_doc1) if w == term1]
                occurrences_doc1_term2 = [i for i, w in enumerate(words_doc1) if w == term2]
                
                # Find occurrences of each term in document 2
                occurrences_doc2_term1 = [i for i, w in enumerate(words_doc2) if w == term1]
                occurrences_doc2_term2 = [i for i, w in enumerate(words_doc2) if w == term2]

                # Calculate average distance between occurrences in document 1
                avg_distance_doc1 = calculate_avg_distance(occurrences_doc1_term1) + calculate_avg_distance(occurrences_doc1_term2)

                # Calculate average distance between occurrences in document 2
                avg_distance_doc2 = calculate_avg_distance(occurrences_doc2_term1) + calculate_avg_distance(occurrences_doc2_term2)

                # Store the average distances in the dictionary
                avg_distances[(term1, term2)].extend([avg_distance_doc1, avg_distance_doc2])

            # Create a DataFrame from the average distances dictionary
            df = pd.DataFrame(avg_distances, index=['Document 1', 'Document 2']).T
            df.index = pd.MultiIndex.from_tuples(df.index, names=['Term 1', 'Term 2'])
            df.reset_index(inplace=True)
            
            return df
            
    class PROX2:
        
        def explain(query, document1, document2):
            # Tokenize the query sentence into words
            query_words = query.split()

            # Tokenize the documents into words
            words_doc1 = document1.split()
            words_doc1 = [word.replace('.', '') for word in words_doc1]
            words_doc2 = document2.split()
            words_doc2 = [word.replace('.', '') for word in words_doc2]

            # Initialize a dictionary to store the position of the first occurrence of each query term
            first_occurrences = {}

            # Find the position of the first occurrence of each query term in document 1
            for term in query_words:
                index = next((i for i, word in enumerate(words_doc1) if word == term), None)
                first_occurrences[f'Document 1 - {term}'] = index

            # Find the position of the first occurrence of each query term in document 2
            for term in query_words:
                index = next((i for i, word in enumerate(words_doc2) if word == term), None)
                first_occurrences[f'Document 2 - {term}'] = index

            # Print the positions of the first occurrence of each query term
            for key, value in first_occurrences.items():
                print(f"{key}: {value}")

    class PROX3:
    
        def explain(query, document1, document2):
            # Find the first occurrence of the entire query string in document 1
            index_doc1 = document1.find(query)
            
            # Find the first occurrence of the entire query string in document 2
            index_doc2 = document2.find(query)
            
            # Print the first occurrence of the entire query string in both documents
            print(f"First occurrence of the query in Document 1: {index_doc1 if index_doc1 != -1 else 'Not present'}")
            print(f"First occurrence of the query in Document 2: {index_doc2 if index_doc2 != -1 else 'Not present'}")

    
           
        