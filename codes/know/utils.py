from collections import Counter


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


def params_freeze(model, para_update="klg"):
    """
    Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    
    print("=" * 20 + "all parameters name" + "=" * 20)
    params = []
    for name, param in model.named_parameters():
        params.append(name)
        pass
    print(params)

    print("=" * 20 + "require updating name" + "=" * 20)
    params = []
    """Freeze all the parameters except knowledge related module."""
    for name, param in model.named_parameters():
        if para_update not in name:
            param.requires_grad = False
            pass
        else:
            params.append(name)
            pass
        pass
    print(params)

    return model


def save_results(results_file_path, results):
    with open(results_file_path, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}={v}\n")


def save_preds(preds_file_path, ids_preds):
    with open(preds_file_path, 'w') as f:
        ids, preds = ids_preds
        for id, pred in zip(ids, preds):
            f.write(f"{id},{pred}\n")
