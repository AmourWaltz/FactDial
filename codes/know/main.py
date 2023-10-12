import os
import time
from pdb import set_trace
from argparse import ArgumentParser

import torch
from transformers.models.bart.modeling_bart import *
from transformers import (set_seed, GPT2Config, GPT2Tokenizer)
from torch import distributed as dist

from data import *
from train import *
from model import *
from utils import *

"""
Parameter settings.
"""
arg_parser = ArgumentParser()
"""Model options."""
arg_parser.add_argument('--batch_size', type=int, default=2)
arg_parser.add_argument('--learning_rate', type=float, default=6e-5)
arg_parser.add_argument('--num_epoch', type=int, default=20)
arg_parser.add_argument('--seq_len', type=int, default=512)
arg_parser.add_argument('--model_path', type=str, default="gpt")
arg_parser.add_argument('--load_path', type=str, default='./exp/batch7_epoch4_seq512_lr6e-05')

"""Data options."""
arg_parser.add_argument('--data_path', type=str, default=None)
arg_parser.add_argument('--cache_path', type=str, default=None, help='Efficiently load data from cached files.')

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

"""Knowledge module options."""
arg_parser.add_argument('--know_type', type=str, default="nkb")
arg_parser.add_argument('--know_size', type=int, default=1024)

args = arg_parser.parse_args()


if __name__ == '__main__':
    """Display time."""
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
    global_rank = dist.get_rank()
    world_size = torch.distributed.get_world_size()

    """
    Info: Load GPT2 model, tokenizer and dataset. 
    """
    args.model_path = args.load_path.replace("models/", "").replace("exp/", "")

    config = GPT2Config.from_json_file(os.path.join(args.load_path, 'config.json'))
    tokenizer = GPT2Tokenizer.from_pretrained(args.load_path, do_lower_case=True)
    config.know_type = args.know_type
    config.know_size = args.know_size

    """Add special tokens and resize the embedding size; Debug by running GPT2 on CPU for more readable.""" 
    special_tokens = {'pad_token':'<|pad|>', 'mask_token': '[MASK]'}
    # print(len(tokenizer))
    tokenizer.add_special_tokens(special_tokens)

    model = SelfGPT2LMHeadModel.from_pretrained(args.load_path, config=config, tokenizer=tokenizer)
    """Resize the word embedding size after adding special tokens."""
    # print(len(tokenizer))
    # set_trace()
    model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()

    model = params_freeze(model)

    """Set all related module using CUDA to CPU, and set off DDP."""
    # device = 'cpu'
    model.to(device)
    # set_trace()
    
    train(args, model, tokenizer)
