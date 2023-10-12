import os
import math
import time
from tqdm import tqdm
from pdb import set_trace

import torch
from transformers.models.bart.modeling_bart import *
from transformers import (AdamW, get_linear_schedule_with_warmup)
from torch.nn.parallel import DistributedDataParallel as DDP
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data import *
from utils import *
from model import *


"""
Info: Training setting part.
Note: Including DDP, optimizer, etc.
"""
def train(args, model, tokenizer):
    """Handling dataset."""
    import logging
    logger = logging.getLogger(__file__)

    train_loader, valid_loader = klg_loader(args, tokenizer, logger)
    train_steps = len(train_loader) * args.num_epoch
    
    save_path = f'{args.exp_path}/{args.model_type}_batch{args.batch_size}_epoch{args.num_epoch}_seq{args.seq_len}_lr{args.learning_rate}_{args.know_type}_dim{args.know_size}'
    
    print("\nSave path: {}.".format(save_path))

    if args.global_rank==0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            pass
        pass

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    """Optimizer setting."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
         'weight_decay': float(args.weight_decay)},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=float(args.learning_rate))
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.warmup_steps, 
                                                num_training_steps=train_steps)
    """
    Info: Training process part.
    """
    best_accu, loss_sum, valid_f1, valid_bleu = None, 0., 0., 0.

    """tqdm: prints a dynamically updating progressbar."""
    with tqdm(total=train_steps) as t:
        for epoch in range(args.num_epoch):
            model.train()
            start_time = time.time()
            for _, batch in enumerate(train_loader):
                t.set_description("train and valid")
                """Output loss."""
                input_ids, token_ids, pos_ids, label_ids, loss_mask = get_klg_batch(batch, tokenizer, device=args.device)
                # set_trace()
                losses, _, _ = model(input_ids=input_ids, token_type_ids=token_ids, position_ids=pos_ids, labels=label_ids, return_dict=False)
                loss_mask = loss_mask.view(-1)
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum() 
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                loss_sum += loss.item()
                
                if args.global_rank == 0:
                    t.set_postfix(ppl=math.exp(loss))
                    t.update(1)
                    pass
                pass

            """Evaluation process on validation set and select the best model."""
            model.eval()

            with open(os.path.join(save_path, "epoch" + str(epoch)) + ".json",'w', encoding='utf-8') as f:
                with torch.no_grad():
                    for batch in valid_loader:
                        input_ids, token_ids, pos_ids, label_ids, loss_mask = get_klg_batch(batch, tokenizer, device=args.device)

                        """Apply model.generate to decode the outputs."""
                        # set_trace()
                        label_mask = torch.ne(label_ids[..., 1:], tokenizer.pad_token_id)
                        label_ids = torch.masked_select(label_ids[..., 1:], label_mask)
                        inputs = input_ids[..., :-(len(label_ids) - 1)]
                        # print(inputs, input_ids, label_ids)
                        outputs = model.module.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id, 
                                                        return_dict_in_generate=True, output_scores=True)
                        output_ids, _ = outputs[0], outputs[-1]
                        
                        """Store knowledge sentence, masked knowledge entities and predicted entities."""
                        klg_sent = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                        ent_pre = tokenizer.decode(output_ids[0][inputs.size(-1)-1:], skip_special_tokens=False)
                        ent_mask = tokenizer.decode(label_ids, skip_special_tokens=False)

                        """Calculate BLUE and F1 between response and reference."""
                        reply_f1 = f1_overlap(ent_pre, ent_mask)
                        reply_bleu = sentence_bleu([ent_pre], ent_mask, smoothing_function=SmoothingFunction().method7, weights=[1./4, 1./4, 1./4, 1./4])
                        valid_f1 += reply_f1
                        valid_bleu += reply_bleu
                        
                        res_data = {
                            "knowledge": "".join(klg_sent),
                            "masked_entity": "".join(ent_mask),
                            "predicted_entity": "".join(ent_pre),
                            "reply_f1": round(reply_f1, 4),
                            "reply_bleu": round(reply_bleu, 4)
                        }
                        json.dump(obj=res_data, fp=f, ensure_ascii=False)
                        f.write("\n")
                        pass
                    pass
                
                new_bleu = round(valid_bleu / len(valid_loader), 4)

                """Save the model if validation loss is the best we've seen so far."""
                if not best_accu or new_bleu < best_accu:
                    model_save = (model.module if hasattr(model, "module") else model)
                    model_save.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    best_accu = new_bleu
                    pass

                """Record each training epoch results."""
                epoch_log = {}

                train_loss = loss_sum / len(train_loader)
                elapsed = time.time() - start_time

                epoch_log = {
                    "epoch": epoch,
                    "time-ms/batch": round(elapsed * 1000 / len(train_loader), 3),
                    "train_ppl": round(math.exp(train_loss), 3),
                    "valid_bleu": round(valid_bleu / len(valid_loader), 4),
                    "valid_f1": round(valid_f1 / len(valid_loader), 4)
                }


                for key, value in epoch_log.items():
                    f.write(f'{key}: {value}\n')
                    pass
                pass
            pass
        pass
    pass

