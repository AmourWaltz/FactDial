import re
from collections import Counter
from pdb import set_trace


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


def klg_params_freeze(args, model, para_update="klg"):
    """
    Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """

    params = []
    for name, param in model.named_parameters():
        params.append(name)
        pass
    
    # if args.global_rank == 0:
    #     print("=" * 20 + "All parameters name" + "=" * 20)
    #     print(params)

    params = []
    """Freeze all the parameters of knowledge related module."""
    for name, param in model.named_parameters():
        if para_update in name:
            param.requires_grad = False
            pass
        else:
            params.append(name)
            pass
        pass

    # if args.global_rank == 0:
    #     print("\n" + "=" * 20 + "Model Parameters Required Updating" + "=" * 20)
    #     print(params)

    return model


def params_update(args, model):
    """
    Print parameters that need to be updates or param.require_grad == True.
    Parameter efficient strategy is adopted: For GPT2-Large, only the parameters in top-50% (>18) layers are optimized.
                                             For GPT2-XL, only the parameters in top-33% layers are optimized.
    """

    params_bp, params_nbp = [], []
    parambp_sum = 0
    
    if args.model_path == "gpt2-m":
        layer_thread = 12
        pass
    elif args.model_path == "gpt2-l":
        layer_thread = 9
        pass
    elif args.model_path == "gpt2-xl":
        layer_thread = 24
        pass
    else:
        layer_thread = 0

    for name, param in model.named_parameters():
        """Extract the layer id of parameter name.
           Example: from "transformer.h.34.attn.c_proj.weight" get 34.
        """
        layer = re.findall(r"\d+", name)
        if len(layer) > 0:
            layer = int(layer[0])
            if layer >= layer_thread:
                params_bp.append(name)
                parambp_sum += param.numel()
            else:
                param.requires_grad = False
                params_nbp.append(name)
                pass
            pass
        elif "ln_f" in name:
            params_bp.append(name)
            parambp_sum += param.numel()
            pass
        else:
            param.requires_grad = False
            params_nbp.append(name)
            pass
        # set_trace()
    
    if args.global_rank == 0:
        print("\n" + "=" * 20 + " Required Updating Model Parameter List. In Total: {:d} ".format(parambp_sum) + "=" * 20)
        print(params_bp)

        # print("\n" + "=" * 20 + "Model Parameters Freezed" + "=" * 20)
        # print(params_nbp) if params_nbp else None
        
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
