import os
import json

from utils import *


def wiki_collect(data_path):
    files = os.listdir(data_path)
    wiki_list = [0 for _ in range(len(files))]

    for file in files:
        inp_file = os.path.join(data_path, file)
        with open(inp_file) as fr:
            data = json.load(fr)
            wiki_list[data["wikiDocumentIdx"]] = data
            # print(data)
            # break

    # print(len(wiki_list))
    return wiki_list


def wiki_split(wiki_list):
    klg_list = []
    for wiki_para in wiki_list:
        # key words: "0", "1", "2", "3", "wikiDocumentIdx"
        # Process wiki_para["0"]: ['cast', 'critical_response', 'director', 'genre', 'introduction', 'movieName', 'rating', 'year']
        klg_snips = []
        for sent in wiki_para["0"]["cast"]:
            klg_snips.append(sent)
        
        # print(len(wiki_para["0"]["critical_response"]))
        for sent in wiki_para["0"]["critical_response"]:
            klg_snips.append(sent)

        # print(wiki_para["0"]["director"])
        klg_snips.append("The director is " + wiki_para["0"]["director"])

        # print(wiki_para["0"]["genre"])
        klg_snips.append("The movie's genre is " + wiki_para["0"]["genre"])

        # print(wiki_para["0"]["introduction"].split(". "))
        klg_snips.extend(wiki_para["0"]["introduction"].split(". "))

        # print(wiki_para["0"]["movieName"])
        klg_snips.append("The movie's name is " + wiki_para["0"]["movieName"])

        # print(wiki_para["0"]["rating"])
        klg_snips.append("The movie's rating is " + ", ".join(wiki_para["0"]["rating"]))

        # print(wiki_para["0"]["year"])
        klg_snips.append("The movie's year is " + wiki_para["0"]["year"])

        # Process wiki_para ["1", "2", "3"]
        # print(wiki_para["1"].split(". "))
        for sent in wiki_para["1"].split(". "):
            if sent.strip() != "":
                klg_snips.append(sent)

        for sent in wiki_para["2"].split(". "):
            if sent.strip() != "":
                klg_snips.append(sent)

        for sent in wiki_para["3"].split(". "):
            if sent.strip() != "":
                klg_snips.append(sent)

        # print(klg_snips)
        # print(len(klg_snips))

        klg_list.append(klg_snips)
    
    # print(len(klg_list))
    return klg_list


def dial_collect(data_path, klg_list):
    for dataset in ["train", "valid", "test"]:
        files = os.listdir(os.path.join(inp_path, dataset))
        ins_set = []
        for file in files:
            with open(os.path.join(inp_path, dataset, file)) as fr:
                data = json.load(fr)

            # Process data keys(): ['date', 'history', 'rating', 'status', 'uid1LogInTime', 'uid1LogOutTime', 'uid1response', 
            # 'uid2LogInTime', 'uid2LogOutTime', 'uid2response', 'user1_id', 'user2_id', 'whoSawDoc', 'wikiDocumentIdx']
            # print(data["history"])
            # print(data["whoSawDoc"])
            saw_docs = data['whoSawDoc']
            wiki_para = klg_list[data["wikiDocumentIdx"]]
            sents = []
            for utter in data["history"]:
                # utter keys(): ['docIdx', 'text', 'uid', 'utcTimestamp']
                # print(utter["text"])
                sents.append(utter["text"])
                if utter["uid"] in saw_docs:
                    # print(utter["uid"], utter["docIdx"])
                    if len(utter["text"]) == 0:
                        continue
                    if len(sents) > 2:
                        score_list = []
                        for wiki_sent in wiki_para:
                            if len(wiki_sent) == 0:
                                continue
                            f1_score = f1_overlap(utter["text"], wiki_sent)
                            score_list.append(f1_score)
                            # print(f1_score)
                            # break
                        if max(score_list) > 0.7:
                            gold_klg = wiki_para[score_list.index(max(score_list))]
                            # print(wiki_para[score_list.index(max(score_list))], max(score_list), utter["text"])
                            ins = {
                                "context": sents[-3:-1],
                                "knowledge": gold_klg,
                                "response": utter["text"]
                            }
                            ins_set.append(ins)
                        # print(utter["docIdx"])
                        # print(len(wiki_para))
                        # break
                        # sents
                # break
            # break
        print("The number of samples in {} set: {}.".format(dataset, len(ins_set)))
        with open(os.path.join(inp_path, dataset + ".json"), "w") as fw:
            json.dump(obj=ins_set, fp=fw, ensure_ascii=False, indent=4)
        # break

if __name__=="__main__":
    data_path = "./data/cmudog/WikiData"
    wiki_list = wiki_collect(data_path=data_path)
    klg_list = wiki_split(wiki_list=wiki_list)

    inp_path = "./data/cmudog/Conversations"
    dial_collect(inp_path, klg_list)



