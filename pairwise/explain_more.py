import itertools
import pandas as pd

def calculate_avg_distance(occurrences):
    distances = [occurrences[i+1] - occurrences[i] for i in range(len(occurrences)-1)]
    return sum(distances) / len(distances) if distances else 0

class explain_more:

    class TFC1:

          def explain(query, document1, document2):

            if abs(len(document1) - len(document2)) >= 0.1 * min(len(document1),len(document2)):
                print("Lengths of documents not similar")
                return 0

            def term_frequency(term, document):
              return document.split().count(term)

            query_terms = query.split()

            doc1_tf = sum(term_frequency(term, doc1) for term in query_terms)
            doc2_tf = sum(term_frequency(term, doc2) for term in query_terms)

            print(f"Term Frequency of query terms in document1 is {doc1_tf}")
            print(f"Term Frequency of query terms in document2 is {doc2_tf}")

    class PROX1:

          def calculate_avg_distance(occurrences):
              if len(occurrences) < 2:
                  return float('inf')  
              distances = [occurrences[i + 1] - occurrences[i] for i in range(len(occurrences) - 1)]
              return sum(distances) / len(distances)

          def explain(query, document1, document2):
              query_words = query.split()

              words_doc1 = document1.split()
              words_doc1 = [word.replace('.', '') for word in words_doc1]
              words_doc2 = document2.split()
              words_doc2 = [word.replace('.', '') for word in words_doc2]

              term_pairs = list(itertools.combinations(query_words, 2))

              avg_distances = {pair: [] for pair in term_pairs}
              term_frequencies = {word: {'Document 1': 0, 'Document 2': 0} for word in query_words}

              for term1, term2 in term_pairs:
                  occurrences_doc1_term1 = [i for i, w in enumerate(words_doc1) if w == term1]
                  occurrences_doc1_term2 = [i for i, w in enumerate(words_doc1) if w == term2]

                  occurrences_doc2_term1 = [i for i, w in enumerate(words_doc2) if w == term1]
                  occurrences_doc2_term2 = [i for i, w in enumerate(words_doc2) if w == term2]

                  avg_distance_doc1 = calculate_avg_distance(occurrences_doc1_term1) + calculate_avg_distance(occurrences_doc1_term2)
                  avg_distance_doc2 = calculate_avg_distance(occurrences_doc2_term1) + calculate_avg_distance(occurrences_doc2_term2)

                  avg_distances[(term1, term2)].extend([avg_distance_doc1, avg_distance_doc2])

              for word in query_words:
                  term_frequencies[word]['Document 1'] = words_doc1.count(word)
                  term_frequencies[word]['Document 2'] = words_doc2.count(word)

              rows = []
              for word in query_words:
                  row = [f'tf({word})', term_frequencies[word]['Document 1'], term_frequencies[word]['Document 2']]
                  rows.append(row)

              total_avg_dist_doc1 = 0
              total_avg_dist_doc2 = 0

              for term_pair, distances in avg_distances.items():
                  row = [f'avg_dist({term_pair[0]}, {term_pair[1]})', distances[0], distances[1]]
                  rows.append(row)
                  total_avg_dist_doc1 += distances[0]
                  total_avg_dist_doc2 += distances[1]

              num_pairs = len(term_pairs)
              rows.append(['num pairs', num_pairs, num_pairs])
              rows.append(['Total_avg_dist', total_avg_dist_doc1 / num_pairs, total_avg_dist_doc2 / num_pairs])

              df = pd.DataFrame(rows, columns=['Metric', 'Document 1', 'Document 2'])

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

    class PROX4:

        def explain(query, doc1, doc2):
          query_terms = set(query.split())

          def smallest_span(document):
              words = document.split()
              term_positions = {term: [] for term in query_terms}

              for idx, word in enumerate(words):
                  if word in query_terms:
                      term_positions[word].append(idx)

              min_span_length = float('inf')
              min_span_non_query_count = float('inf')
              min_span = []

              for term in term_positions:
                  for start_pos in term_positions[term]:
                      end_pos = start_pos
                      for other_term in term_positions:
                          if other_term != term:
                              closest_pos = min(term_positions[other_term], key=lambda x: abs(x - start_pos))
                              end_pos = max(end_pos, closest_pos)

                      if end_pos - start_pos + 1 < min_span_length:
                          min_span = words[start_pos:end_pos + 1]
                          min_span_length = len(min_span)
                          min_span_non_query_count = sum(1 for word in min_span if word not in query_terms)

              return min_span_non_query_count

          def calculate_gap(document):
              min_span_non_query_count = smallest_span(document)
              words = document.split()
              gap_frequency = words.count(str(min_span_non_query_count))

              return (min_span_non_query_count, gap_frequency)

          gap1 = calculate_gap(doc1)
          gap2 = calculate_gap(doc2)

          print(f"ω(doc1, query) = {gap1}")
          print(f"ω(doc2, query) = {gap2}")

    class PROX5:

        def explain(query, doc1, doc2):

          query_terms = query.split()

          def find_positions(term, document):
              positions = []
              words = document.split()
              for idx, word in enumerate(words):
                  if word == term:
                      positions.append(idx)
              return positions

          def smallest_span_around(term_positions, all_positions, num_terms):
              min_span = float('inf')
              for pos in term_positions:
                  spans = []
                  for i in range(num_terms):
                      term_pos = all_positions[i]
                      if term_pos:
                          distances = [abs(pos - p) for p in term_pos]
                          spans.append(min(distances))
                  if len(spans) == num_terms:
                      min_span = min(min_span, max(spans) - min(spans) + 1)
              return min_span

          def average_smallest_span(document):
              all_positions = [find_positions(term, document) for term in query_terms]
              total_span = 0
              count = 0

              for i, term_positions in enumerate(all_positions):
                  if term_positions:
                      span = smallest_span_around(term_positions, all_positions, len(query_terms))
                      if span < float('inf'):
                          total_span += span
                          count += 1

              return total_span / count if count > 0 else float('inf')

          span1 = average_smallest_span(doc1)
          span2 = average_smallest_span(doc2)

          print(f"average smallest text span across all occurrences query terms in doc1 is {span1}")
          print(f"average smallest text span across all occurrences query terms in doc2 is {span2}")

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

    class STMC1:

        def explain(query, document1, document2):

          similarity_doc1, similarity_doc2 = wordnet_similarity(query, document1, document2)

          print(f"similarity score of doc1 with query terms is {similarity_doc1}")
          print(f"similarity score of doc2 with query terms is {similarity_doc2}")

    class AND:

        def explain(query, document1, document2):

            query_terms = set(query.lower().split())
            doc1_terms = set(document1.lower().split())
            doc2_terms = set(document2.lower().split())

            if query_terms.issubset(doc1_terms):
                print("All query terms present in document 1")
            if query_terms.issubset(doc2_terms):
                print("All query terms present in document 2")

            not_in_doc1 = query_terms - doc1_terms
            if not_in_doc1:
                print("Query terms not present in document 1:", not_in_doc1)
            else:
                print("All query terms are present in document 1")

            not_in_doc2 = query_terms - doc2_terms
            if not_in_doc2:
                print("Query terms not present in document 2:", not_in_doc2)
            else:
                print("All query terms are present in document 2")

    class DIV:

        def explain(query, document1, document2):

            query_terms = set(query.lower().split())
            doc1_terms = set(document1.lower().split())
            doc2_terms = set(document2.lower().split())

            jaccard_coefficient_doc1 = len(query_terms.intersection(doc1_terms)) / len(query_terms.union(doc1_terms))
            jaccard_coefficient_doc2 = len(query_terms.intersection(doc2_terms)) / len(query_terms.union(doc2_terms))

            print(f"Jaccard Co-efficient of doc1 is:{jaccard_coefficient_doc1}")
            print(f"Jaccard Co-efficient of doc2 is:{jaccard_coefficient_doc2}")
