import os
import math
import json
import logging
from tqdm import tqdm
from pdb import set_trace

import torch
from rouge import Rouge
from transformers.models.bart.modeling_bart import *
from torch.nn.parallel import DistributedDataParallel as DDP
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from codes.dial.data import *
from model import *
from utils import *

"""
Info: Test part, print the generated response and calculate factual consistency metrics.
"""
def infer(args, model, tokenizer):
    """Handling dataset."""
    import logging
    logger = logging.getLogger(__file__)

    [test_loader] = build_loaders(args, tokenizer, logger)
    eval_steps = len(test_loader)
    
    save_path = f"{args.exp_path}/{args.data}/{args.model_path}/beam{args.num_beam}_samp{args.sample}_temp{args.temp}_tk{args.top_k}_tp{args.top_p}"

    print("\nSave path: {}.\n".format(save_path))

    if args.global_rank==0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            pass
        pass

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    loss_sum, reply_bleu_sum, reply_f1_sum, klg_f1_sum, reply_rouge_sum = 0., 0., 0., 0., 0.
    rouger = Rouge()
    
    with tqdm(total=eval_steps) as t:
        with open(os.path.join(save_path, "test.json"), 'w', encoding='utf-8') as f:
            with torch.no_grad():
                for idx, batch in enumerate(test_loader):
                    input_ids, _, _, label_ids, klg_ids, _ = get_batch(batch, tokenizer, device=args.device)

                    """Apply model.generate to decode the outputs."""
                    # set_trace()
                    label_mask = torch.ne(label_ids[..., 1:], tokenizer.pad_token_id)
                    label_ids = torch.masked_select(label_ids[..., 1:], label_mask)
                    inputs = input_ids[..., :-(len(label_ids) - 1)]
                    outputs = model.module.generate(inputs, do_sample=args.sample, max_length=1000, num_beams=args.num_beam, 
                                                    pad_token_id=tokenizer.eos_token_id, temperature=args.temp, top_k=args.top_k, 
                                                    top_p=args.top_p, return_dict_in_generate=True, output_scores=True)
                    output_ids, _ = outputs[0], outputs[-1]
                    # loss2_sum += loss

                    """Calculate BLUE and NLI metrics for factual consistency."""
                    hist = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    klg = tokenizer.decode(klg_ids[0], skip_special_tokens=False)
                    reply = tokenizer.decode(output_ids[0][inputs.size(-1)-1:], skip_special_tokens=False)
                    refer = tokenizer.decode(label_ids, skip_special_tokens=False)
                    
                    """Calculate BLUE and F1-overlap score metrics for factual consistency."""
                    # set_trace()
                    reply_f1, klg_f1 = f1_overlap(reply, refer), f1_overlap(reply, klg)
                    reply_bleu = sentence_bleu([refer], reply, smoothing_function=SmoothingFunction().method7, weights=[1./4, 1./4, 1./4, 1./4])
                    reply_rouge = rouger.get_scores(reply, refer)[0]["rouge-l"]["f"]
                    
                    reply_bleu_sum += reply_bleu
                    reply_f1_sum += reply_f1
                    klg_f1_sum += klg_f1
                    reply_rouge_sum += reply_rouge

                    res_data = {
                        "context": "".join(hist),
                        "response": "".join(reply),
                        "knowledge": "".join(klg),
                        "reference": "".join(refer),
                        "klg_f1": round(klg_f1, 4),
                        "reply_f1": round(reply_f1, 4),
                        "reply_bleu": round(reply_bleu, 4),
                        "reply_rouge": round(reply_rouge, 4)
                    }
                    json.dump(obj=res_data, fp=f, ensure_ascii=False)
                    f.write("\n")

                    t.set_postfix(kf_rf_rb_rr=[round(klg_f1_sum / (idx + 1), 3), round(reply_f1_sum / (idx + 1), 3), round(reply_bleu_sum / (idx + 1), 3), round(reply_rouge_sum / (idx + 1), 3)])
                    t.update(1)
                    pass
                pass
            pass
        
        with open(os.path.join(save_path, "test.txt"), 'w', encoding='utf-8') as f:
            f.write("Overall: ppl: {:.4f} reply_f1: {:.4f} | klg_f1: {:.4f} | reply_bleu: {:.4f} | reply_rouge: {:.4f}\n\n".format(
                    math.exp(loss_sum / len(test_loader)), reply_f1_sum / len(test_loader), klg_f1_sum / len(test_loader), 
                    reply_bleu_sum / len(test_loader), reply_rouge_sum / len(test_loader)))
            pass
        pass
    pass


"""
Info: Inference part, chatting with the model.
"""
def human_infer(args, model, tokenizer, num_turn=4):
    for step in range(num_turn):
        # Encode the new user input, add the eos_token and return a tensor in Pytorch.
        user_input_ids = tokenizer.encode(input(">> User: ") + tokenizer.eos_token, return_tensors='pt').to(args.device)
        
        # Append the new user input tokens to the chat history.
        bot_input_ids = torch.cat([chat_hist_ids, user_input_ids], dim=-1) if step > 0 else user_input_ids
        
        # Generated a response while limiting the total chat history to 1000 tokens.
        chat_hist_ids = model.generate(bot_input_ids, max_length=1000, num_beams=args.num_beam, 
                                    pad_token_id=tokenizer.eos_token_id, temperature=args.temp, top_k=args.top_k, 
                                    top_p=args.top_p, return_dict_in_generate=True, output_scores=True)
        
        # Pretty print last ouput tokens from bot.
        # set_trace()
        print("DialoGPT: {}".format(tokenizer.decode(chat_hist_ids["sequences"][:, user_input_ids.shape[-1]:][0], skip_special_tokens=False)))
        pass
    pass

