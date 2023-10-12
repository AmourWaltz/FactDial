import os
import random
import spacy
import json, csv
from pdb import set_trace
from collections import Counter


##############################################################################
# Info： Collect knowledge. 
##############################################################################

def klg_making():
    inp_path = "./../../../../data/wizard_of_wikipedia/dialog_only/"
    out_path = "./../../../../data/wizard_of_wikipedia/klg_only/"

    inp_files = ["train.json", "valid.json", "test.json"]

    nlp = spacy.load("en_core_web_sm")
    
    klgs = []
    for file in inp_files:
        with open(os.path.join(inp_path, file), "r", encoding="utf-8") as f:
            dialogs = json.loads(f.read())
            for dial in dialogs:
                for turn in dial:
                    # set_trace()
                    if turn["klg"] is None:
                        continue
                    if "no_passages_used" in turn["klg"].keys():
                        continue
                    
                    reply = turn["utter"]
                    
                    for klg in turn["klg"].values():
                        klg_ents = nlp(klg).ents
                        reply_ents = nlp(reply).ents

                        for ent in klg_ents:
                            if str(ent) in str(reply_ents):
                                mask_klg = klg.replace(str(ent), "[MASK]")
                                mask_ent = str(ent)
                                new_samp = {
                                    "knowledge": mask_klg,
                                    "masked_entity": mask_ent
                                }
                                klgs.append(new_samp)
                                pass
                            pass
                        pass
                    pass
                pass
            pass
    
        print("Number of {:s} knowledges {:d}".format(file[:-5], len(klgs)))
    
    with open(os.path.join(out_path, "train.json"), "w", encoding="utf-8") as f:
        json.dump(klgs, f, ensure_ascii="utf-8")
        pass
    pass


if __name__=="__main__":
    klg_making()

