class pairwise:

  def __init__(self, query, doc1, doc2, index_path):
        self.query = query
        self.doc1 = doc1
        self.doc2 = doc2
        self.index_path = index_path

  def explain(self, query, doc1, doc2, axiom_classes):
        results = {'Query': query, 'Document 1': doc1[:25] + '.....', 'Document 2': doc2[:25] + "....."}

        for axiom in axiom_classes:
            if isinstance(axiom, str):
                axiom_instance = self._get_axiom_class(axiom)
            else:
                axiom_instance = axiom

            if axiom_instance:
                combined_score = axiom_instance.compare(query, doc1, doc2)

                if combined_score > 0:
                    result = 1
                elif combined_score < 0:
                    result = -1
                else:
                    result = 0

                results[axiom_instance.__class__.__name__] = result

        df = pd.DataFrame([results])
        return df

  def evaluate_axiom_expression(self, expression, query, doc1, doc2):
    elements = expression.split()
    score = 0
    current_op = '+'
    current_coeff = 1

    def apply_operation(axiom_score):
        nonlocal score
        if current_op == '+':
            score += current_coeff * axiom_score
        elif current_op == '-':
            score -= current_coeff * axiom_score

    i = 0
    while i < len(elements):
        element = elements[i]

        if element in ['+', '-']:
            current_op = element
            current_coeff = 1
        elif '*' in element or '/' in element:
            parts = element.split('*')
            if len(parts) == 2:
                coeff, axiom_name = parts
                current_coeff = int(coeff)
                axiom = self._get_axiom_class(axiom_name)
                if axiom:
                    axiom_score = axiom.compare(query, doc1, doc2)
                    apply_operation(axiom_score)
            else:
                parts = element.split('/')
                if len(parts) == 2:
                    coeff, axiom_name = parts
                    current_coeff = 1 / int(coeff)
                    axiom = self._get_axiom_class(axiom_name)
                    if axiom:
                        axiom_score = axiom.compare(query, doc1, doc2)
                        apply_operation(axiom_score)
        else:
            current_coeff = 1
            axiom = self._get_axiom_class(element)
            if axiom:
                axiom_score = axiom.compare(query, doc1, doc2)
                apply_operation(axiom_score)

        i += 1

    return score

  def explain_details(self, query, doc1, doc2, axiom_name):
        axiom_class = getattr(explain_more, axiom_name, None)

        if axiom_class:
            explanation_df = axiom_class.explain(query, doc1, doc2)
            return explanation_df
        else:
            return "Axiom not found in explain_more class"

  def _get_axiom_class(self, axiom_name):
        axiom_classes_mapping = {
            "TFC1": self.TFC1(),
            "TFC3": self.TFC3(self.index_path),
            "PROX1": self.PROX1(),
            "PROX2": self.PROX2(),
            "PROX3": self.PROX3(),
            "PROX4": self.PROX4(),
            "PROX5": self.PROX5(),
            "LNC1": self.LNC1(),
            "LNC2": self.LNC2(),
            "TF_LNC": self.TF_LNC(),
            "LB1": self.LB1(),
            "STMC1": self.STMC1(),
            "AND": self.AND(),
            "REG": self.REG(),
            "DIV": self.DIV()
        }
        return axiom_classes_mapping.get(axiom_name)

  class TFC1:

    def compare(self,query, document1, document2):

        if abs(len(document1) - len(document2)) >= 0.1 * min(len(document1),len(document2)):
          return 0

        def term_frequency(term, document):
          return document.split().count(term)

        query_terms = query.split()

        doc1_tf = sum(term_frequency(term, doc1) for term in query_terms)
        doc2_tf = sum(term_frequency(term, doc2) for term in query_terms)

        if doc1_tf > doc2_tf:
            return 1
        elif doc1_tf == doc2_tf:
            return 0
        else:
            return -1

  class TFC3:

        def __init__(self, index_path):
            self.term_discrimination_values = self.calculate_term_discrimination_values(index_path)

        def calculate_term_discrimination_values(self, index_path):
            term_doc_freq = {}
            total_docs = 0

            for filename in os.listdir(index_path):
                if filename.endswith('.txt'):
                    total_docs += 1
                    with open(os.path.join(index_path, filename), 'r') as file:
                        document = file.read()
                        terms = set(document.split())
                        for term in terms:
                            if term in term_doc_freq:
                                term_doc_freq[term] += 1
                            else:
                                term_doc_freq[term] = 1

            term_discrimination_values = {term: 1.0 / freq for term, freq in term_doc_freq.items()}
            return term_discrimination_values

        def compare(self, query, document1, document2):
            query_words = set(query.split())
            document1_words = set(document1.split())
            document2_words = set(document2.split())

            total_occurrences1 = sum(document1.split().count(term) for term in query_words)
            total_occurrences2 = sum(document2.split().count(term) for term in query_words)

            term_discrimination_values = {term: self.term_discrimination_values.get(term, 1.0) for term in query_words}

            score1 = total_occurrences1 * sum(term_discrimination_values[term] for term in query_words)
            score2 = total_occurrences2 * sum(term_discrimination_values[term] for term in query_words)

            if score1 > score2:
                return 1
            elif score2 > score1:
                return -1
            else:
                distinct_terms1 = len(document1_words.intersection(query_words))
                distinct_terms2 = len(document2_words.intersection(query_words))

                if distinct_terms1 > distinct_terms2:
                    return 1
                elif distinct_terms2 > distinct_terms1:
                    return -1
                else:
                    return 0


  class PROX1:

    def compare(self,query, document1, document2):
      query_words = query.split()

      if not all(word in document1 for word in query_words) or not all(word in document2 for word in query_words):
          return 0

      words_doc1 = document1.split()
      words_doc1 = [word.replace('.', '') for word in words_doc1]
      words_doc2 = document2.split()
      words_doc2 = [word.replace('.', '') for word in words_doc2]

      common_word_pairs = [(word1, word2) for word1 in query_words if word1 in words_doc1 and word1 in words_doc2
                          for word2 in query_words if word2 in words_doc1 and word2 in words_doc2 and word1 != word2]

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

      sum_words_between_doc1 = sum(words_between_pairs_doc1.values())
      sum_words_between_doc2 = sum(words_between_pairs_doc2.values())

      total_possible_pairs = len(query_words) * (len(query_words) - 1) // 2

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
      query_words = query.split()

      if not all(word in document1 for word in query_words) or not all(word in document2 for word in query_words):
          return 0

      words_doc1 = document1.split()
      words_doc2 = document2.split()
      first_positions_doc1 = [words_doc1.index(word) if word in words_doc1 else None for word in query_words]
      first_positions_doc2 = [words_doc2.index(word) if word in words_doc2 else None for word in query_words]

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
      if query in document1 and query in document2:
          first_position_doc1 = document1.find(query)
          first_position_doc2 = document2.find(query)

          if first_position_doc1 < first_position_doc2:
              return 1
          elif first_position_doc1 > first_position_doc2:
              return -1
          else:
              return 0
      elif query in document1:
          return 1
      elif query in document2:
          return -1
      else:
          return 0

  class PROX4:

    def compare(self,query, doc1, doc2):
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

      if gap1 < gap2:
          return 1
      elif gap1 > gap2:
          return -1
      else:
          return 0


  class PROX5:

    def compare(self,query, doc1, doc2):

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

      if span1 < span2:
          return 1
      elif span1 > span2:
          return -1
      else:
          return 0


  class LNC1:

    def compare(self,query, doc1, doc2):

      query_words = query.split()

      count_query_terms_doc1 = sum(1 for word in query_words if word in doc1)
      count_query_terms_doc2 = sum(1 for word in query_words if word in doc2)

      max_allowed_difference = 0.1 * min(count_query_terms_doc1, count_query_terms_doc2)
      if abs(count_query_terms_doc1 - count_query_terms_doc2) > max_allowed_difference:
          return 0
      else:
          if len(doc1) == len(doc2):
              return 0
          elif len(doc1) < len(doc2):
              return 1
          else:
              return -1

  class LNC2:

    def compare(self,query, doc1, doc2):
      original_doc, copied_doc = (doc1, doc2) if len(doc1) <= len(doc2) else (doc2, doc1)
      original_words = set(original_doc.split())
      copied_words = set(copied_doc.split())

      jaccard_coefficient = len(original_words.intersection(copied_words)) / len(original_words)

      if jaccard_coefficient >= 0.8:
          shared_terms = original_words.intersection(copied_words)
          min_frequency = min(original_doc.split().count(term) for term in shared_terms)
          max_frequency = max(original_doc.split().count(term) for term in shared_terms)
          m = max(1, min_frequency / max_frequency)
          return 1 if len(original_doc) <= len(copied_doc) else -1
      else:
          return 0

  class TF_LNC:

      def compare(self,query, document1, document2):

          query_words = set(query.split())
          document1_words = set(document1.split())
          document2_words = set(document2.split())

          doc1_tf = query_words.intersection(document1_words)
          doc2_tf = query_words.intersection(document2_words)

          words1 = document1.split()
          words2 = document2.split()

          filtered_words1 = [word for word in words1 if word not in doc1_tf]
          filtered_words2 = [word for word in words2 if word not in doc2_tf]

          new_doc1 = ' '.join(filtered_words1)
          new_doc2 = ' '.join(filtered_words2)

          max_len = max(len(new_doc1), len(new_doc2))
          tolerance = 0.1 * max_len

          if abs(len(new_doc1) - len(new_doc2)) > tolerance:
              return 0
          else:
              if doc1_tf > doc2_tf:
                return -1
              if doc1_tf < doc2_tf:
                return 1
              if doc1_tf == doc1_tf:
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

      most_similar_term = get_most_similar_term(query_terms)

      if not most_similar_term:
          return 0
      all_texts = [query] + [document1, document2]

      vectorizer = CountVectorizer()
      term_frequency_matrix = vectorizer.fit_transform(all_texts)

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
