import os
import time
from tqdm import tqdm
from pdb import set_trace

import torch
from transformers.models.bart.modeling_bart import *
from transformers import (AdamW, get_linear_schedule_with_warmup)
from torch.nn.parallel import DistributedDataParallel as DDP

from data import *
from tune import *
from model import *
from utils import *


"""
Info: Training setting part.
Note: Including DDP, optimizer, etc.
"""
def train(args, model, tokenizer):
    """Handling dataset."""
    import logging
    logger = logging.getLogger(__file__)

    if args.task == "pretrain":
        [train_loader, valid_loader] = build_loaders(args, tokenizer, logger)
        train_steps = len(train_loader) * args.num_epoch
        pass
    else:
        [train_loader, valid_loader] = snli_build_loaders(args, tokenizer, logger)
        train_steps = len(train_loader) * args.num_epoch
        pass
    
    save_path = f'{args.exp_path}/{args.model_type}_batch{args.batch_size}_epoch{args.num_epoch}_seq{args.seq_len}_lr{args.learning_rate}_ratio{args.neg_ratio}_{args.task}'

    if args.global_rank==0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(os.path.join(save_path, "log"))
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
    best_accu = 0.

    """tqdm: prints a dynamically updating progressbar."""
    with tqdm(total=train_steps) as t:
        for epoch in range(args.num_epoch):
            model.train()
            num_samp = 0
            train_pos, valid_pos = 0, 0
            start_time = time.time()
            for idx, batch in enumerate(train_loader):
                t.set_description("train and valid")
                """Output loss."""
                input_ids, token_ids, pos_ids, attn_ids, label_ids = get_batch(batch, tokenizer, training=True, device=args.device)
                loss, logits = model(input_ids=input_ids, token_type_ids=token_ids, attention_mask=attn_ids, position_ids=pos_ids, labels=label_ids, return_dict=False)
                loss = loss.view(-1)
                
                """Calculate the accurately classified samples."""
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                if args.task == "fact":
                    # set_trace()
                    if idx % 100 == 0:
                        print(logits[0][0])
                        
                    if args.global_rank == 0:
                        t.set_postfix(logits=logits[0][0].item())
                        t.update(1)
                        pass
                else:
                    output_ids = torch.argmax(logits, -1)
                    train_pos += sum(torch.eq(output_ids, label_ids))
                    num_samp += output_ids.size(-1)

                    # if idx % 100 == 0:
                    #     print(train_pos)
                    #     print(num_samp)

                    if args.global_rank == 0:
                        train_accu = train_pos / (idx * args.batch_size)
                        t.set_postfix(accu=loss.item())
                        t.update(1)
                        pass

                if idx % 100 == 0 and idx != 0:
                    """Evaluation process on validation set and select the best model."""
                    model.eval()

                    with open(os.path.join(save_path, "log/epoch" + str(epoch)) + ".json",'w', encoding='utf-8') as f:
                        with torch.no_grad():
                            for batch in valid_loader:
                                """Output loss."""
                                input_ids, token_ids, pos_ids, attn_ids, label_ids = get_batch(batch, tokenizer, training=True, device=args.device)
                                loss, logits = model(input_ids=input_ids, token_type_ids=token_ids, attention_mask=attn_ids, position_ids=pos_ids, labels=label_ids, return_dict=False)

                                if args.task == "fact":
                                    valid_pos += logits[0][0]
                                    pass
                                else:
                                    output_ids = torch.argmax(logits, -1)
                                    valid_pos += torch.eq(output_ids, label_ids)
                                    pass
                                pass
                            pass

                        valid_accu = valid_pos / len(valid_loader)

                        """Save the model if validation loss is the best we've seen so far."""
                        if not best_accu or best_accu < valid_accu:
                            model_save = (model.module if hasattr(model, "module") else model)
                            model_save.save_pretrained(save_path)
                            tokenizer.save_pretrained(save_path)
                            best_accu = valid_accu
                            pass

                        """Record each training epoch results."""
                        epoch_log = {}

                        elapsed = time.time() - start_time

                        epoch_log = {
                            "epoch": epoch,
                            "time-ms/batch": round(elapsed * 1000 / len(train_loader), 3),
                            "train_accu": train_accu if args.task != "fact" else None,
                            "valid_accu": valid_accu
                        }

                        for key, value in epoch_log.items():
                            f.write(f'{key}: {value}\n')
                            pass
                        pass
                    pass
                pass
            pass
        pass
    pass

