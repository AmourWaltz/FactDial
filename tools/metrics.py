from utils.utils import ngrams
from collections import Counter

__all__ = ["distinct_n_sentence_level", "distinct_n_corpus_level", "f1_overlap"]


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    sentence = sentence.split(" ")
    if len(sentence) == 0:
        print ("error len(sentence) == 0")
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


def f1_overlap(candidate, reference, value="F1"):
    """
    F1 token-level overlap of a candidate and a reference.
    """
    common = Counter(candidate) & Counter(reference)
    overlap = sum(common.values())
    recall, precision = overlap / len(reference), overlap / len(candidate)

    if value == "P":
        return precision
    elif value == "R":
        return recall
    else:
        return (2 * recall * precision) / (recall + precision + 1e-12)


class TfIdf:
    """
    tfidf calculation.
    """
    def __init__(self):
        self.weighted = False
        self.documents = []
        self.corpus_dict = {}

    def add_document(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
            self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0

        # normalizing the dictionary
        length = float(len(list_of_words))
        for k in doc_dict:
            doc_dict[k] = doc_dict[k] / length

        # add the normalized document to the corpus
        self.documents.append([doc_name, doc_dict])

    def similarities(self, list_of_words):
        """
        Returns a list of all the [docname, similarity_score] pairs relative to a list of words.
        """

        # building the query dictionary
        query_dict = {}
        for w in list_of_words:
            query_dict[w] = query_dict.get(w, 0.0) + 1.0

        # normalizing the query
        length = float(len(list_of_words))
        for k in query_dict:
            query_dict[k] = query_dict[k] / length

        # computing the list of similarities
        sims = []
        for doc in self.documents:
            score = 0.0
            doc_dict = doc[1]
            for k in query_dict:
                if k in doc_dict:
                    score += (query_dict[k] / self.corpus_dict[k]) + (doc_dict[k] / self.corpus_dict[k])
            sims.append([doc[0], score])

        return sims


print(distinct_n_sentence_level("可能 吧 , 我 觉得 我 很 喜欢 笑 。",1))