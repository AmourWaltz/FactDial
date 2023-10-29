import os
import time
import math
import json
from tqdm import tqdm
from pdb import set_trace

import torch
from torch import distributed as dist
from transformers.models.bart.modeling_bart import *
from transformers import (AdamW, get_linear_schedule_with_warmup)
from torch.nn.parallel import DistributedDataParallel as DDP
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from codes.dial.data import *
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

    [train_loader, valid_loader] = build_loaders(args, tokenizer, logger)
    train_steps = len(train_loader) * args.num_epoch
    
    save_path = f'{args.exp_path}/{args.data}/{args.model_path}/batch{args.batch_size}_epoch{args.num_epoch}_seq{args.seq_len}_lr{args.learning_rate}_{args.gpu_num}gpu'

    if args.global_rank==0:
        print("\nSave path: {}.\n".format(save_path))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            pass
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
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
    best_loss = 0.

    """tqdm: prints a dynamically updating progressbar."""
    with tqdm(total=train_steps) as t:
        for epoch in range(args.num_epoch):
            model.train()
            loss_sum, train_bleu, train_f1, loss2_sum, valid_bleu, valid_f1 = 0., 0., 0., 0., 0., 0.
            start_time = time.time()
            for batch in train_loader:
                t.set_description("train and valid")
                """Output loss."""
                # set_trace()
                input_ids, token_ids, pos_ids, label_ids, _, loss_mask = get_batch(batch, tokenizer, 
                                                                                   training=True, 
                                                                                   seq_len=args.seq_len, 
                                                                                   device=device)
                
                losses, logits, _ = model(input_ids=input_ids, 
                                          token_type_ids=token_ids, 
                                          position_ids=pos_ids, 
                                          labels=label_ids, 
                                          return_dict=False)
                
                loss_mask = loss_mask.view(-1)
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()

                loss_sum += loss.item()

                """Use tokens logits to calculate token-level."""
                outputs = torch.argmax(logits, -1)
                label_mask = torch.ne(input_ids[..., 1:], tokenizer.pad_token_id)
                labels = torch.masked_select(label_ids[..., 1:], label_mask)
                outputs = torch.masked_select(outputs, label_mask)
                
                """Calculate BLUE and NLI metrics for factual consistency."""
                # hist = [tokenizer.decode(x).strip() for x in input_ids[0]]
                # klg = [tokenizer.decode(x).strip() for x in klg_ids[0]]
                # reply = [tokenizer.decode(x).strip() for x in outputs]
                # refer = [tokenizer.decode(x).strip() for x in labels]
                
                reply = tokenizer.decode(outputs[..., -(len(labels) - 1):], skip_special_tokens=False)
                refer = tokenizer.decode(labels, skip_special_tokens=False)

                """Calculate BLUE and F1-overlap score metrics for factual consistency."""
                reply_f1 = f1_overlap(reply, refer)
                reply_bleu = sentence_bleu([refer], reply, smoothing_function=SmoothingFunction().method7, weights=[1./4, 1./4, 1./4, 1./4])
                train_f1 += reply_f1
                train_bleu += reply_bleu

                if dist.get_rank() == 0:
                    t.set_postfix(ppl=math.exp(loss.item()))
                    t.update(1)
                    pass
                pass

            """Evaluation process on validation set and select the best model."""
            model.eval()

            with open(os.path.join(save_path, "epoch" + str(epoch)) + ".json",'w', encoding='utf-8') as f:
                with torch.no_grad():
                    for batch in valid_loader:
                        input_ids, token_ids, pos_ids, label_ids, klg_ids, loss_mask = get_batch(batch, tokenizer, 
                                                                                                 device=device)                        
                        """Apply model.generate to decode the outputs."""
                        # set_trace()
                        label_mask = torch.ne(label_ids[..., 1:], tokenizer.pad_token_id)
                        label_ids = torch.masked_select(label_ids[..., 1:], label_mask)
                        inputs = input_ids[..., :-(len(label_ids) - 1)]
                        outputs = model.module.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id, 
                                                        return_dict_in_generate=True, output_scores=True)
                        output_ids, logits = outputs[0], outputs[-1]
                        # loss2_sum += loss

                        hist = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                        klg = tokenizer.decode(klg_ids[0], skip_special_tokens=False)
                        reply = tokenizer.decode(output_ids[0][inputs.size(-1)-1:], skip_special_tokens=False)
                        refer = tokenizer.decode(label_ids, skip_special_tokens=False)
                        
                        """Calculate BLUE and F1 between response and reference."""
                        reply_f1 = f1_overlap(reply, refer)
                        reply_bleu = sentence_bleu([refer], reply, smoothing_function=SmoothingFunction().method7, weights=[1./4, 1./4, 1./4, 1./4])
                        valid_f1 += reply_f1
                        valid_bleu += reply_bleu
                        
                        """Calculate BLUE and F1 between response and knowledge."""
                        # klg_f1 = f1_overlap(reply, klg)
                        # klg_bleu = corpus_bleu([klg], reply, smoothing_function=SmoothingFunction().method7, weights=[1. / 1])
                        # valid_f1 += klg_f1
                        # valid_bleu += klg_bleu

                        res_data = {
                            "context": "".join(hist),
                            "response": "".join(reply),
                            "knowledge": "".join(klg),
                            "reference": "".join(refer),
                            "reply_f1": round(reply_f1, 4),
                            "reply_bleu": round(reply_bleu, 4)
                        }
                        json.dump(obj=res_data, fp=f, ensure_ascii=False)
                        f.write("\n")
                        pass
                    pass

                valid_loss = loss2_sum / len(valid_loader)

                """Save the model if validation loss is the best we've seen so far."""
                if not best_loss or valid_loss < best_loss:
                    if args.global_rank == 0:
                        model_save = (model.module if hasattr(model, "module") else model)
                        model_save.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        best_loss = valid_loss
                        pass

                """Record each training epoch results."""
                epoch_log = {}

                train_loss = loss_sum / len(train_loader)
                elapsed = time.time() - start_time

                epoch_log = {
                    "epoch": epoch,
                    "time-ms/batch": round(elapsed * 1000 / len(train_loader), 3),
                    "train_ppl": round(math.exp(train_loss), 3),
                    "valid_ppl": round(math.exp(valid_loss), 3),
                    "train_bleu": round(train_bleu / len(train_loader), 4),
                    "train_f1": round(train_f1 / len(train_loader), 4),
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

