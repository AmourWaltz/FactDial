import os
import json
from pdb import set_trace

# inp_path = "./../../../../data/wizard_of_wikipedia/release"
inp_path = "./../../../../data/wizard_of_wikipedia/dialog_only/"
out_path = "./../../../../data/wizard_of_wikipedia/dialog_only/"

# inp_files = ["train.json", "valid_1.json", "valid_2.json", "test_1.json", "test_2.json"]
inp_files = ["train.json", "valid.json", "test.json"]

##############################################################################
# Info： Collect the dialogues containing "speaker", "text" and 
#        "checked_sentence" from the released WoW data. 
#        Regard "Apprentice" as "user" and "Wizard" as "bot".
##############################################################################

def dialog_prep():
    for file in inp_files:
        dialogs = []
        with open(os.path.join(inp_path, file), "r") as f:
            # set_trace()
            dataset = json.loads(f.read())
            pass
        
        for data in dataset:
            new_dialog = []
            # set_trace()
            for turn in data["dialog"]:
                if "Wizard" in turn["speaker"]:
                    new_turn = {
                        "role": "bot",
                        "utter": turn["text"],
                        "klg": turn["checked_sentence"]
                    }
                    pass
                elif "Apprentice" in turn["speaker"]:
                    new_turn = {
                        "role": "user",
                        "utter": turn["text"],
                        "klg": None
                    }
                    pass
                
                new_dialog.append(new_turn)
                pass
            
            dialogs.append(new_dialog)
            pass
        
        with open(os.path.join(out_path, file), "w") as f:
            json.dump(dialogs, f, ensure_ascii=False)
            pass
        pass
    pass


##############################################################################
# Info： Make training data of dialogues containing "context", "response" and 
#        "knowledge" for training process. 
##############################################################################

def sample_making(dialog):
    utt_hist = []
    samps = []
    for turn in dialog:
        utt_hist.append(turn["utter"])
        if turn["role"] == "bot" and len(utt_hist) > 1:
            samp = {
                "context": utt_hist[:-1],
                "response": utt_hist[-1],
                "knowledge": turn["klg"]
            }
            assert turn["klg"] is not None
            samps.append(samp)
        pass
    
    return samps


def dataset_making():
    for file in inp_files:
        samp_pool = []
        with open(os.path.join(inp_path, file), "r") as f:
            # set_trace()
            dataset = json.loads(f.read())
            for dialog in dataset:
                samp_pool.extend(sample_making(dialog))
                pass
            pass
        
        print("Number of samples in " + file[:-5] + " set: {:d}.".format(len(samp_pool)))
        
        with open(os.path.join(out_path, file), "w") as f:
            json.dump(samp_pool, f, ensure_ascii=False)
            pass
        pass
    pass


##############################################################################
# Info： Collect knowledge. 
##############################################################################

def klg_making():
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
                    
                    for klg in turn["klg"].values():
                        klgs.append(klg)
                        pass
                    pass
                pass
            pass
        pass
    
    print("Number of knowledges {:d}".format(len(klgs)))
    
    with open(os.path.join(out_path, "klg.json"), "w", encoding="utf-8") as f:
        json.dump(klgs, f, ensure_ascii="utf-8")
        pass
    pass

if __name__=="__main__":
    klg_making()

