class pairwise:

  def __init__(self, query, doc1, doc2, index_path):
        self.query = query
        self.doc1 = doc1
        self.doc2 = doc2
        self.index_path = index_path

  def explain(self, axiom_classes):
        for axiom_class in axiom_classes:
            result = axiom_class.compare(self.query, self.doc1, self.doc2)
            print(f"Result for {axiom_class.__class__.__name__}: {result}")
          
  def _get_axiom_class(self, axiom_name):
        # Implement logic to retrieve the axiom class based on its name
        axiom_classes_mapping = {
            "TFC1": self.TFC1(),
            "TFC3": self.TFC3(),
            "PROX1": self.PROX1(),
            "PROX2": self.PROX2(),
            "PROX3": self.PROX3(),
            "PROX4": self.PROX4(),
            "PROX5": self.PROX5(),
            "LNC1": self.LNC1(),
            "LNC2": self.LNC2(),
            "LB1": self.LB1(),
            "STMC1": self.STMC1(),
            "AND": self.AND(),
            "REG": self.REG(),
            "DIV": self.DIV()
        }
        return axiom_classes_mapping.get(axiom_name)
          
  class TFC1:

    def compare(self,query, document1, document2):
        # Convert the documents and query into sets of words
        query_words = set(query.split())
        document1_words = set(document1.split())
        document2_words = set(document2.split())

        # Calculate the number of common words between the query and each document
        common_words1 = len(query_words.intersection(document1_words)) / len(document1_words)
        common_words2 = len(query_words.intersection(document2_words)) / len(document2_words)

        # Check if the normalized common words are approximately equal
        if abs(common_words1 - common_words2) <= 0.1 * min(common_words1, common_words2):
            return 0

        # returns the document with most frequent terms (normalized) 1 means the first document whereas -1 means the second document
        if common_words1 > common_words2:
            return 1
        else:
            return -1

  class TFC3:

    def compare(self,query, document1, document2, term_discrimination_values):
      # Convert the documents and query into sets of words
      query_words = set(query.split())
      document1_words = set(document1.split())
      document2_words = set(document2.split())

      # Calculate the total occurrences of query terms in each document
      total_occurrences1 = sum(document1.split().count(term) for term in query_words)
      total_occurrences2 = sum(document2.split().count(term) for term in query_words)

      # Calculate the term discrimination value for each query term
      term_discrimination_values = {term: term_discrimination_values.get(term, 1.0) for term in query_words}

      # Calculate the score based on total occurrences and term discrimination values
      score1 = total_occurrences1 * sum(term_discrimination_values[term] for term in query_words)
      score2 = total_occurrences2 * sum(term_discrimination_values[term] for term in query_words)

      # Compare the scores and return the document with a higher score
      if score1 > score2:
          return 1
      elif score2 > score1:
          return -1
      else:
          # If scores are equal, compare based on the number of distinct query terms
          distinct_terms1 = len(document1_words.intersection(query_words))
          distinct_terms2 = len(document2_words.intersection(query_words))

          # Return the document with more distinct query terms
          if distinct_terms1 > distinct_terms2:
              return 1
          elif distinct_terms2 > distinct_terms1:
              return -1
          else:
              return 0  # If everything is equal, return 0

  class PROX1:

    def compare(self,query, document1, document2):
      # Tokenize the query sentence into words
      query_words = query.split()

      if not all(word in document1 for word in query_words) or not all(word in document2 for word in query_words):
          return 0

      # Tokenize the documents into words
      words_doc1 = document1.split()
      words_doc1 = [word.replace('.', '') for word in words_doc1]
      words_doc2 = document2.split()
      words_doc2 = [word.replace('.', '') for word in words_doc2]

      # Find all pairs of words in the query that are present in both documents
      common_word_pairs = [(word1, word2) for word1 in query_words if word1 in words_doc1 and word1 in words_doc2
                          for word2 in query_words if word2 in words_doc1 and word2 in words_doc2 and word1 != word2]

      # Calculate the number of words between each pair in each document
      words_between_pairs_doc1 = {}
      words_between_pairs_doc2 = {}

      for word1, word2 in common_word_pairs:
          indices_query_doc1 = [i for i, word in enumerate(words_doc1) if word == word1 or word == word2]
          indices_query_doc2 = [i for i, word in enumerate(words_doc2) if word == word1 or word == word2]

          if len(indices_query_doc1) == 2:
              start, end = min(indices_query_doc1), max(indices_query_doc1)
              words_between_pairs_doc1[(word1, word2)] = abs(end - start) - 1

          if len(indices_query_doc2) == 2:
              start, end = min(indices_query_doc2), max(indices_query_doc2)
              words_between_pairs_doc2[(word1, word2)] = abs(end - start) - 1

      # Calculate the sum of words between all pairs for each document
      sum_words_between_doc1 = sum(words_between_pairs_doc1.values())
      sum_words_between_doc2 = sum(words_between_pairs_doc2.values())

      # Calculate the total number of possible query pairs
      total_possible_pairs = len(query_words) * (len(query_words) - 1) // 2  # nC2 combination

      # Calculate the ratio of the sum of words between all pairs to the total number of possible query pairs
      ratio_doc1 = sum_words_between_doc1 / total_possible_pairs if total_possible_pairs > 0 else 0
      ratio_doc2 = sum_words_between_doc2 / total_possible_pairs if total_possible_pairs > 0 else 0

      if ratio_doc1 < ratio_doc2:
          return 1
      elif ratio_doc1 > ratio_doc2:
          return -1
      else:
          return 0

  class PROX2:

    def compare(self,query, document1, document2):
      # Tokenize the query sentence into words
      query_words = query.split()

      if not all(word in document1 for word in query_words) or not all(word in document2 for word in query_words):
          return 0

      # Tokenize the documents into words
      words_doc1 = document1.split()
      words_doc2 = document2.split()

      # Find the first position of each query term in both documents
      first_positions_doc1 = [words_doc1.index(word) if word in words_doc1 else None for word in query_words]
      first_positions_doc2 = [words_doc2.index(word) if word in words_doc2 else None for word in query_words]

      # Calculate the sum of the first positions for each document, ignoring None values
      sum_first_positions_doc1 = sum(position for position in first_positions_doc1 if position is not None)
      sum_first_positions_doc2 = sum(position for position in first_positions_doc2 if position is not None)

      if sum_first_positions_doc1 < sum_first_positions_doc2:
        return 1
      elif sum_first_positions_doc1 > sum_first_positions_doc2:
        return -1
      else:
        return 0

  class PROX3:

    def compare(self,query, document1, document2):
      # Check if the entire query phrase appears in both documents
      if query in document1 and query in document2:
          # Find the first position of the query phrase in both documents
          first_position_doc1 = document1.find(query)
          first_position_doc2 = document2.find(query)

          # Compare and return the result
          if first_position_doc1 < first_position_doc2:
              return 1
          elif first_position_doc1 > first_position_doc2:
              return -1
          else:
              return 0
      elif query in document1:
          # Query is only present in Document 1
          return 1
      elif query in document2:
          # Query is only present in Document 2
          return -1
      else:
          # Query is not present in either document
          return 0

  class PROX4:

    def compare(self,query, document1, document2):
      pass

  class PROX5:
  
    def compare(self,query, document1, document2):
      pass

  class LNC1:
  
    def compare(self,query, doc1, doc2):
      # Tokenize the query sentence into words
      query_words = query.split()

      # Count the number of query terms in each document
      count_query_terms_doc1 = sum(1 for word in query_words if word in doc1)
      count_query_terms_doc2 = sum(1 for word in query_words if word in doc2)

      # Compare the number of query terms and document lengths
      max_allowed_difference = 0.1 * min(count_query_terms_doc1, count_query_terms_doc2)
      if abs(count_query_terms_doc1 - count_query_terms_doc2) > max_allowed_difference:
          return 0
      else:
          # Compare the lengths
          if len(doc1) == len(doc2):
              return 0
          elif len(doc1) < len(doc2):
              return 1
          else:
              return -1

  class LNC2:

    def compare(self,query, doc1, doc2):
      # Determine the original and copied documents based on length
      original_doc, copied_doc = (doc1, doc2) if len(doc1) <= len(doc2) else (doc2, doc1)

      # Tokenize documents into words
      original_words = set(original_doc.split())
      copied_words = set(copied_doc.split())

      # Calculate the Jaccard coefficient of the documents' vocabularies
      jaccard_coefficient = len(original_words.intersection(copied_words)) / len(original_words)

      # Check if the Jaccard coefficient is at least 80%
      if jaccard_coefficient >= 0.8:
          # Calculate the ratio of minimum and maximum term frequencies of shared terms
          shared_terms = original_words.intersection(copied_words)
          min_frequency = min(original_doc.split().count(term) for term in shared_terms)
          max_frequency = max(original_doc.split().count(term) for term in shared_terms)

          # Derive the value of m based on the ratio of minimum and maximum term frequencies
          m = max(1, min_frequency / max_frequency)

          # Determine which document is shorter and return 1 or -1 accordingly
          return 1 if len(original_doc) <= len(copied_doc) else -1
      else:
          # If the Jaccard coefficient is less than 80%, return 0
          return 0

    class TF_LNC:
      
      def compare(self,query, doc1, doc2):
        # Check if the query is a single word
        if " " in query:
            return 0
  
        # Count the number of occurrences of the query term in each document
        count_query_terms_doc1 = doc1.split().count(query)
        count_query_terms_doc2 = doc2.split().count(query)
  
        # Check if the lengths of both documents are the same within a 10% difference
        max_allowed_length_difference = 0.1 * min(len(doc1), len(doc2))
        if abs(len(doc1) - len(doc2)) <= max_allowed_length_difference:
            # Return the document with the higher occurrence of the query term
            if count_query_terms_doc1 > count_query_terms_doc2:
                return 1
            elif count_query_terms_doc1 < count_query_terms_doc2:
                return -1
            else:
                # If both documents have the same occurrence, return 0
                return 0
        else:
            # If the length difference is outside the 10% range, return 0
            return 0

  class LB1:

    def compare(self,query, document1, document2):
      query_terms = set(query.lower().split())
      doc1_terms = set(document1.lower().split())
      doc2_terms = set(document2.lower().split())

      unique_to_doc1 = [term for term in query_terms if term in doc1_terms and term not in doc2_terms]
      unique_to_doc2 = [term for term in query_terms if term in doc2_terms and term not in doc1_terms]

      if unique_to_doc1 == unique_to_doc2:
        return 0
      if unique_to_doc1 > unique_to_doc2:
        return 1
      return -1

  class STMC1:

    def compare(self,query, document1, document2):
      similarity_doc1, similarity_doc2 = wordnet_similarity(query, document1, document2)

      # Compare the similarity scores and return 1, -1, or 0
      if similarity_doc1 > similarity_doc2:
          return 1
      elif similarity_doc1 < similarity_doc2:
          return -1
      else:
          return 0

  class AND:

    def compare(self,query, document1, document2):
      query_terms = set(query.lower().split())
      doc1_terms = set(document1.lower().split())
      doc2_terms = set(document2.lower().split())

      if query_terms.issubset(doc1_terms):
          return 1
      elif query_terms.issubset(doc2_terms):
          return -1
      else:
          return 0

  class REG:

    def compare(self,query, document1, document2):
      query_terms = query.lower().split()

      # Find the query term with the highest average similarity to other query terms
      most_similar_term = get_most_similar_term(query_terms)

      if not most_similar_term:
          return 0
      #print(most_similar_term)

      # Combine query and documents for term frequency analysis
      all_texts = [query] + [document1, document2]

      # Calculate term frequency using CountVectorizer
      vectorizer = CountVectorizer()
      term_frequency_matrix = vectorizer.fit_transform(all_texts)

      # Find the document with the highest term frequency for the most similar term
      most_similar_term_index = vectorizer.vocabulary_[most_similar_term]
      doc1_term_frequency = term_frequency_matrix[-2, most_similar_term_index]
      doc2_term_frequency = term_frequency_matrix[-1, most_similar_term_index]

      if doc1_term_frequency > doc2_term_frequency:
          return 1
      elif doc2_term_frequency > doc1_term_frequency:
          return -1
      else:
          return 0

  class DIV:

    def compare(self,query, document1, document2):
      query_terms = set(query.lower().split())
      doc1_terms = set(document1.lower().split())
      doc2_terms = set(document2.lower().split())

      jaccard_coefficient_doc1 = len(query_terms.intersection(doc1_terms)) / len(query_terms.union(doc1_terms))
      jaccard_coefficient_doc2 = len(query_terms.intersection(doc2_terms)) / len(query_terms.union(doc2_terms))

      if jaccard_coefficient_doc1 < jaccard_coefficient_doc2:
          return 1
      else:
          return -1
      return 0
