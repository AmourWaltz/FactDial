import os
import random
import spacy
import json, csv
from pdb import set_trace
from collections import Counter


##############################################################################
# Info： Preparing training set containing positive and negetive samples.
##############################################################################
def f1_overlap(candidate, reference, value="F1"):
    common = Counter(candidate) & Counter(reference)
    overlap = sum(common.values())
    recall, precision = overlap / len(reference), overlap / len(candidate)

    if value == "P":
        return precision
    elif value == "R":
        return recall
    else:
        return (2 * recall * precision) / (recall + precision + 1e-12)
    
    
def ents_swap(nlp, reply, klg):
    # set_trace()
    swap_reply = " "
    klg_ents = nlp(klg).ents
    reply_ents = nlp(reply).ents
    
    for ent in reply_ents:
        if str(ent) in str(klg_ents):
            # """50% swap the common entity, 50% replace the entity with another in knowledge."""
            # if random.random() > 0.5 and len(klg_ents) > 3:
            #     swap_reply = reply.replace(str(ent), "")
            #     pass
            # else:
            #     # while True:
            #     #     rep_ent = klg_ents[random.randint(0, len(klg_ents) - 1)]
            #     #     if str(rep_ent) != str(ent):
            #     #         break
            #     #     pass
            #     swap_reply = reply.replace(str(ent), str(rep_ent))
            #     pass
            # pass
            swap_reply = reply.replace(str(ent), "")
            pass
        else:
            """50% swap the common entity, 50% replace the entity with another in knowledge."""
            if random.random() > 0.5:
                swap_reply = reply.replace(str(ent), "")
                pass
            else:
                swap_reply = swap_reply
                pass
            pass
        pass
    
    """Find all types of entities in documents."""
    # labels = set([w.label_ for w in document.ents])
    # for label in labels:
    #     entities = [cleanup(e.string, lower=False) for e in document.ents if label==e.label_]
    #     entities = list(set(entities))
    #     print label,entities
    
    """If no common entity mentioned in knowledge, then remove all the entities in response."""
    if swap_reply == " ":
        for ent in reply_ents:
            swap_reply = reply.replace(str(ent), "")
            pass
        pass
    
    """Randomly cut the reply."""
    rand_num = random.random()
    if rand_num > 0.9:
        reply_list = swap_reply.split()
        swap_reply = " ".join(reply_list[:int(len(reply_list)/2)])
        pass
    elif rand_num < 0.1:
        reply_list = swap_reply.split()
        swap_reply = " ".join(reply_list[int(len(reply_list)/2):])
        pass

    """Avoid empty swapped reply candidate."""
    if len(reply_ents) == 0 or swap_reply == " ":
        reply_list = reply.split()
        swap_reply = " ".join(reply_list[:int(len(reply_list)/2)])
        pass
    
    return swap_reply


def data_builder(dials):
    dataset = []
    f1_gold, f1_swap = 0, 0
    nlp = spacy.load("en_core_web_sm")
    
    for idx, dial in enumerate(dials):
        data = {}
        context = dial["context"][0]
        gold_reply = dial["response"]
        gold_klg = dial["knowledge"]
        """
        Three negetive sample making strategies:
        1. Random pairing: randomly select a response from other context.
        2. Negation: negation applied to response to untouch the knowledge.
        3. Entity swapping: mask the commen entity mentioned in response and knowledge.
        """
        while True:
            """A response is randomly selected."""
            rand_reply = dials[random.randint(0, len(dials) - 1)]["response"]
            if rand_reply != gold_reply:
                break
            pass
        
        """NER to mask the commen entity mentioned in response and knowledge."""
        swap_reply = ents_swap(nlp, gold_reply, gold_klg)
        if len(gold_klg) == 0 or len(gold_reply) == 0 or len(swap_reply) == 0:
            continue
        f1_gold += f1_overlap(gold_reply, gold_klg)
        f1_swap += f1_overlap(swap_reply, gold_klg)
        
        # print(idx, gold_reply)
        # print(idx, swap_reply)
        if idx % 1000 == 0 and idx != 0:
            print("F1 score of gold {:.3f} and entity swapped {:.3f} replies.".format(f1_gold / idx, f1_swap / idx))
            pass
        
        data = {
            "context": context,
            "knowledge": gold_klg,
            "gold_reply": gold_reply,
            "rand_reply": rand_reply,
            "swap_reply": swap_reply
        }
        # print(data)
        dataset.append(data)
    
    print("F1 score of gold {:.3f} and entity swapped {:.3f} replies.".format(f1_gold / len(dials), f1_swap / len(dials)))

    return dataset 


def wiki_process():
    inp_file = "./../../../../../data/wizard_of_wikipedia/dialog_only/"
    out_file = "./../../../../../data/wizard_of_wikipedia/conv_fever/"

    """Merge the inputed data sets."""
    for mode in ["train", "valid", "test"]:
        data_pool = []
        """Building random response pool."""
        with open(os.path.join(inp_file, mode + ".json"), encoding="utf-8") as f:
            dials = json.loads(f.read())
            
            for dial in dials: 
                dial_hist = []
                for turn in dial:
                    utter = turn["utter"]
                    dial_hist.extend([utter])
                    if turn["role"] == "bot" and len(dial_hist) > 1:
                        """Only the reply that have a dependent external knowledge can be built."""
                        if "no_passages_used" in turn["klg"].keys():
                            continue
                        if len(turn["klg"].values()) == 0:
                            continue
        
                        klgs = [klg for klg in turn["klg"].values()]
                        
                        new_data = {
                            "context": dial_hist[:-1],
                            "knowledge": klgs[0],
                            "response": dial_hist[-1]
                        }
                        data_pool.append(new_data)
                        pass
                    pass
                pass
            pass
        
        dataset = data_builder(data_pool)

        print("The number of " + mode + " samples: ", len(dataset))
        # import pdb; pdb.set_trace()
        with open(os.path.join(out_file, mode + ".json"), "w") as f:
            json.dump(dataset, f, ensure_ascii=False)
            pass
        pass


##############################################################################
# Info： Process the dataset for factual accuracy, hallucination and 
#        verifiable evaluation. 
##############################################################################

def nli_making():
    """
    Dataset format after reading .csv files:
        0 number.
        1 model_type: describes the GPT2 model size used to generate the response.
        2 decoding: describes the decoding strategy used to generate the response.
        3 context: the dialog context fed into the neural response generator.
        4 knowledge: the knowledge sentence fed into the neural response generator.
        5 response: the generated response from the model.
        6 Avg Factual Correctness: Is the response generated factually accurate with regards to the input knowledge? Took the average of three scores where each score was on a three-point scale: factually incorrect (0), partially correct (0.5), and completely correct (1).
        7 Hallucination: Is the response generated making up more information than what is provided in the conversational context and input knowledge? Took the majority of three scores where each score was either Yes or No.
        8 Verifiable: Does the response need to be verified (y/n).
    """
    samps = []
    num_verif = 0
    inp_path = "./../../../../../data/wizard_of_wikipedia/conv_fever/"
    out_path = "./../../../../../data/wizard_of_wikipedia/conv_fever/"

    inp_files = ["expert.csv", "gt.csv", "knn.csv", "dpr.csv"]
    
    for file in inp_files:
        print("Start Processing " + file)
        with open(os.path.join(inp_path, file), "r", encoding="utf-8") as f:
            data_pool = csv.reader(f)
            for data in data_pool:
                samp = {}
                """Process the context, knowledge and response."""
                hist, klg, reply = data[3], data[4], data[5]
                
                """Process factual correctness."""
                if data[-1] == "y":
                    verif = True
                    # print(data[0])
                    # set_trace()
                    fact_acc = float(data[6])
                    hallu = True if data[7] == "Yes" else False
                    num_verif += 1
                    pass
                else:
                    verif = False
                    fact_acc = None
                    hallu = None
                    pass
                
                samp = {
                    "hist": hist,
                    "klg": klg,
                    "reply": reply,
                    "verif": verif,
                    "fact_acc": fact_acc,
                    "hallu": hallu
                }
                
                samps.append(samp)
                pass
            pass
        pass
    
    print("Number of samples: {:d}; Number of verifiable samples: {:d}.".format(len(samps), num_verif))
    
    with open(os.path.join(out_path, "conv_fever.json"), "w", encoding="utf-8") as f:
        json.dump(samps, f, ensure_ascii=False)
        pass


if __name__=="__main__":
    wiki_process()

