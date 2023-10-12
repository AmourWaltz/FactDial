import os
import math
import json
from argparse import ArgumentParser

from pdb import set_trace
from rouge import Rouge
from bert_score import score
from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from utils import *

"""
Parameter settings.
"""
arg_parser = ArgumentParser()
"""Model options."""
arg_parser.add_argument('--load_path', type=str, default="models/gpt2-m")
arg_parser.add_argument('--data_path', type=str, default="exp/cmu_dog/gpt2-l/batch3_epoch3_seq448_lr6e-05_2gpu/beam5_sampFalse_temp0.0_tk0.0_tp0.0")


def eval(args, rouger, tokenizer):
    with open(os.path.join(args.data_path, "test.json"), "r") as f:
        data_pool = f.readlines()
        data_len = len(data_pool)
        reply_f1, klg_f1, reply_bleu, reply_rouge, bert_score = 0., 0., 0., 0., 0.
        reply_set, klg_set, refer_set = [], [], []
        
        for idx, data in enumerate(data_pool):
            # set_trace()
            data = json.loads(data)
            reply, klg, refer = data["response"], data["knowledge"], data["reference"]
            reply_f1 += f1_overlap(reply, refer)
            klg_f1 += f1_overlap(reply, klg)
            reply_bleu += sentence_bleu([refer], reply, smoothing_function=SmoothingFunction().method7, weights=[1./4, 1./4, 1./4, 1./4])
            reply_rouge += rouger.get_scores(reply, refer)[0]["rouge-l"]["f"]


            P, R, F1 = score([reply], [klg], model_type="/home/ma-user/work/byxue/models/roberta-large", lang="en", verbose=True)
            bert_score += F1.mean().item()

            print("reply_f1: {:.4f} | klg_f1: {:.4f} | reply_bleu: {:.4f} | reply_rouge: {:.4f} | bert_score: {:.4f}\n\n".format(
                reply_f1 / (idx+1), klg_f1 / (idx+1), reply_bleu / (idx+1), reply_rouge / (idx+1), bert_score / (idx+1)))
            # reply_set.append(reply)
            # klg_set.append(klg)
            pass
        pass

    """Bert Score"""
    # for i in range(0, len(data_pool), 8):
    #     P, R, F1 = score(reply_set[i:i+8], klg_set[i:i+8], model_type="/home/ma-user/work/byxue/models/roberta-large", lang="en", verbose=True)
    #     bert_score += F1.mean().item()
    
    print("Overall: reply_f1: {:.4f} | klg_f1: {:.4f} | reply_bleu: {:.4f} | reply_rouge: {:.4f} | bert_score: {:.4f}\n\n".format(
                reply_f1 / data_len, klg_f1 / data_len, reply_bleu / data_len, reply_rouge / data_len, bert_score / data_len))
    
    with open(os.path.join(args.data_path, "test.txt"), "w") as f:
        f.write("Overall: reply_f1: {:.4f} | klg_f1: {:.4f} | reply_bleu: {:.4f} | reply_rouge: {:.4f} | bert_score: {:.4f}\n\n".format(
                reply_f1 / data_len, klg_f1 / data_len, reply_bleu / data_len, reply_rouge / data_len, bert_score / data_len))
           

if __name__ == '__main__':
    args = arg_parser.parse_args()
    
    """GPT2 Tokenizer load."""
    tokenizer = GPT2Tokenizer.from_pretrained(args.load_path, do_lower_case=True)
    special_tokens = {'pad_token':'<|pad|>'}
    tokenizer.add_special_tokens(special_tokens)

    rouger = Rouge()
    
    eval(args, rouger, tokenizer)
    
