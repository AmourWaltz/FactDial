import os
import time
from pdb import set_trace
from argparse import ArgumentParser

import torch
from transformers.models.bart.modeling_bart import *
from transformers import (set_seed, BertTokenizer, BertConfig)
from torch import distributed as dist

from data import *
from tune import *
from train import *
from model import *
from utils import *
from eval import *

"""
Parameter settings.
"""
arg_parser = ArgumentParser()
"""Model options."""
arg_parser.add_argument('--batch_size', type=int, default=1)
arg_parser.add_argument('--learning_rate', type=float, default=6e-5)
arg_parser.add_argument('--num_epoch', type=int, default=20)
arg_parser.add_argument('--seq_len', type=int, default=512)
arg_parser.add_argument('--load_path', type=str, default='./models/bert')
arg_parser.add_argument('--model_type', type=str, default='bert')

"""Data options."""
arg_parser.add_argument('--data_path', type=str, default=None)
arg_parser.add_argument('--cache_path', type=str, default=None, help='Efficiently load data from cached files.')
arg_parser.add_argument('--neg_ratio', type=float, default=0.5)

"""Task options."""
arg_parser.add_argument('--task', type=str, default="verif")

"""Training options."""
arg_parser.add_argument('--stage', type=str, default="train")
arg_parser.add_argument('--max_steps', type=int, default=0)
arg_parser.add_argument('--eval_steps', type=int, default=2)
arg_parser.add_argument('--weight_decay', type=float, default=1e-2)
arg_parser.add_argument('--warmup_steps', type=int, default=0)
arg_parser.add_argument('--exp_path', type=str, default='./exp')

"""Device options."""
arg_parser.add_argument('--device', type=str, default="cuda")
arg_parser.add_argument('--seed', type=int, default=3407)
arg_parser.add_argument('--local_rank', type=int, default=0)
arg_parser.add_argument('--global_rank', type=int, default=0)
arg_parser.add_argument('--world_size', type=int, default=0)

args = arg_parser.parse_args()


if __name__ == '__main__':
    # Display time.
    print("\n" + "=" * 20 + "Time Display" + "=" * 20)
    
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    print("\n" + "=" * 20 + "All Parameters Name" + "=" * 20)
    
    print(args, "\n")
    
    """
    Info: Device settings: random seed, using cuda or not, distributed setting.
    """
    set_seed(args.seed)

    args.num_gpu = torch.cuda.device_count()

    """Obtain local gpu id and set device."""
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    """Obtain device, the following tensor and model are required to be set .to(device)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.local_rank)

    """Generally using nccl back-end to initialize multi-processes."""
    dist.init_process_group(backend='nccl', init_method='env://')
    args.global_rank = dist.get_rank()
    world_size = torch.distributed.get_world_size()

    args.model_type = args.load_path.replace("models/", "")
    args.model_type = args.load_path.replace("exp/", "")

    """
    Info: Load Roberta model, tokenizer and dataset. 
    """
    if args.task != "pretrain" and args.stage == "train":
        assert args.load_path == './exp/nli_bert'

    config = BertConfig.from_json_file(os.path.join(args.load_path, 'config.json'))
    if args.task == "pretrain":
        """For pretrained NLI model, just training a bi-classifier."""
        config.num_labels = 2
        config.problem_type = "single_label_classification"
        pass
    elif args.task == "verif":
        config.num_labels = 2
        config.problem_type = "single_label_classification"
        pass
    elif args.task == "hallu":
        config.num_labels = 2
        config.problem_type = "single_label_classification"
        pass
    elif args.task == "fact":
        config.problem_type = "regression"
        config.num_labels = 1
        pass

    tokenizer = BertTokenizer.from_pretrained(args.load_path, do_lower_case=True)
    model = SelfBertForSequenceClassification.from_pretrained(args.load_path, config=config)

    """Set all related module using CUDA to CPU, and set off DDP."""
    # device = 'cpu'
    model.to(device)
    # set_trace()
    if args.stage == "train":
        train(args, model, tokenizer)
        pass
    else:
        eval(args, model, tokenizer)
        pass
    
