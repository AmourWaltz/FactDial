import os
from tqdm import tqdm
from pdb import set_trace

import torch
from torch.nn import Softmax

from data import *
from tune import *
from model import *
from utils import *
from score import *


def eval(args, model, tokenizer):
    """Handling dataset."""
    import logging
    logger = logging.getLogger(__file__)

    [test_loader] = tune_loaders(args, tokenizer, logger) if args.stage=="tune" else infer_loaders(args, tokenizer, logger)
    eval_steps = len(test_loader)
    
    softmax = nn.Softmax(dim=-1)
    
    save_path = f"{args.exp_path}/{args.model_type}/log" if args.stage != "score" else f"{args.data_path}"
    
    test_pos, loss_sum = 0, 0.
    
    """tqdm: prints a dynamically updating progressbar."""
    with tqdm(total=eval_steps) as t:
        """Evaluation process on validation set and select the best model."""
        model.eval()
        with open(os.path.join(save_path, "test") + ".json", 'w', encoding='utf-8') as f:
            with torch.no_grad():
                for idx, batch in enumerate(test_loader):
                    """Output loss."""
                    input_ids, token_ids, pos_ids, attn_ids, label_ids = get_batch(batch, args.stage, device=args.device)
                    if args.stage == "score":
                        logits = model(input_ids=input_ids, token_type_ids=token_ids, attention_mask=attn_ids, 
                                       position_ids=pos_ids, labels=label_ids, return_dict=False)
                        # set_trace()
                        if args.task == "fact":
                            test_pos += logits[0][0]
                            if args.global_rank == 0:
                                test_loss = test_pos / (idx + 1)
                                test_accu = test_pos / (idx + 1)
                                t.set_postfix(accu=round(test_accu.item(), 3))
                                t.update(1)
                                pass
                            pass
                        else:
                            score = softmax(logits[0])
                            test_pos += sum(score[:, -1])
                            
                            if args.global_rank == 0:
                                test_accu = test_pos / (idx + 1)
                                t.set_postfix(accu=round(test_accu.item(), 3))
                                t.update(1)
                                pass
                            pass
                    else:
                        loss, logits = model(input_ids=input_ids, token_type_ids=token_ids, attention_mask=attn_ids, 
                                            position_ids=pos_ids, labels=label_ids, return_dict=False)

                        if args.task == "fact":
                            # set_trace()
                            test_pos += logits[0][0]
                            loss_sum += loss
                            if args.global_rank == 0:
                                test_loss = loss_sum / (idx + 1)
                                test_accu = test_pos / (idx + 1)
                                t.set_postfix(accu=round(test_accu.item(), 3))
                                t.update(1)
                                pass
                            pass
                        else:
                            output_ids = torch.argmax(logits, -1)
                            test_pos += torch.eq(output_ids, label_ids)

                            if args.global_rank == 0:
                                test_accu = test_pos / (idx + 1)
                                t.set_postfix(accu=round(test_accu.item(), 3))
                                t.update(1)
                                pass
                            pass
                        pass
                    pass
                pass
            pass
            
        with open(os.path.join(save_path, "test.txt"), 'a', encoding='utf-8') as f:
            f.write("Overall: {} NLI score: {:.4f}, loss: {:.4f} \n".format(args.task, test_accu.item(), test_loss.item() if args.task == "fact" else 0.))
            pass
        pass
    pass

