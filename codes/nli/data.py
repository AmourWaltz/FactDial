import os
import json
import torch
import random
from pdb import set_trace
from itertools import chain
from transformers import cached_path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


##############################################################################
# Info： Tokenize and encode the dataset. 
##############################################################################
def encode_data(tokenizer, data_path, data_cache):
    """Cache the processed data when first tokenizing for quick use."""
    data_cache = data_cache + "_" + type(tokenizer).__name__
    if data_cache and os.path.isfile(data_cache):
        print("Load tokenized dataset from cache at %s", data_cache)
        dataset = torch.load(data_cache)
        pass
    else:
        print("Process dataset from {:s}".format(data_path))
        plan_file = cached_path(data_path)
        with open(plan_file, "r", encoding="utf-8") as f:
            gold_klg, rand_klg, swap_klg = [], [], []
            gold_hist, rand_hist, swap_hist = [], [], []
            data_pool = json.loads(f.read())
            
            for data in data_pool:
                # hist_enc = [tokenizer.encode(data["context"])]
                # klg_enc = [tokenizer.encode(data["knowledge"])]
                # gold_enc = [tokenizer.encode(data["gold_reply"])]
                # rand_enc = [tokenizer.encode(data["rand_reply"])]
                # swap_enc = [tokenizer.encode(data["swap_reply"])]
                
                # set_trace()
                """Sentence-pairing: knowledge as prompt, response as the next_sentence."""
                gold_klg_enc = tokenizer(data["knowledge"], data["gold_reply"], return_tensors='pt')
                rand_klg_enc = tokenizer(data["knowledge"], data["rand_reply"], return_tensors='pt')
                swap_klg_enc = tokenizer(data["knowledge"], data["swap_reply"], return_tensors='pt')

                """Sentence-pairing: context as prompt, response as the next_sentence."""
                gold_hist_enc = tokenizer(data["context"], data["gold_reply"], return_tensors='pt')
                rand_hist_enc = tokenizer(data["context"], data["rand_reply"], return_tensors='pt')
                swap_hist_enc = tokenizer(data["context"], data["swap_reply"], return_tensors='pt')

                gold_klg_enc["label"], gold_hist_enc["label"] = 1, 1
                gold_klg.append(gold_klg_enc)
                gold_hist.append(gold_hist_enc)
                
                rand_klg_enc["label"], rand_hist_enc["label"] = 0, 0
                rand_klg.append(rand_klg_enc)
                rand_hist.append(rand_hist_enc)

                swap_klg_enc["label"], swap_hist_enc["label"] = 0, 0
                swap_klg.append(swap_klg_enc)
                swap_hist.append(swap_hist_enc)
                pass
            pass
        
        dataset = [gold_klg, rand_klg, swap_klg, gold_hist, rand_hist, swap_hist]
        print("Tokenize and encode the dataset.")
        torch.save(dataset, data_cache)
        pass

    return dataset


##############################################################################
# Info： Generate a batch with three types.
##############################################################################
def get_batch(batch_next, stage="tune", device="cpu"):
    if stage == "score":
        input_ids, token_ids, attn_ids = batch_next
    else:
        input_ids, token_ids, attn_ids, label_ids = batch_next

    """When evaluating, the sentence is inputted one-by-one in turn."""
    final_pos = torch.arange(input_ids.size(1), dtype=torch.long)
    pos_ids = final_pos.squeeze(0)
    
    """Put all tensor items to device."""
    input_ids = input_ids.to(torch.device(device))
    token_ids = token_ids.to(torch.device(device))
    attn_ids = attn_ids.to(torch.device(device))
    pos_ids = pos_ids.to(torch.device(device))
    
    if stage != "score":
        label_ids = label_ids.to(torch.device(device))
        return  input_ids, token_ids, pos_ids, attn_ids, label_ids
    else:
        return  input_ids, token_ids, pos_ids, attn_ids, None


##############################################################################
# Info： Make training data of dialogues containing "context", "response" and 
#        "knowledge" for training process. 
##############################################################################
class NLIWikipediaDataset():
    def __init__(self, tokenizer, seq_len=512, data_path=None, data_cache=None, neg_ratio=0.5, batch_first=True, is_label=True):
        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.batch_first = batch_first
        self.is_label = is_label

        """Different separate token and token type ids."""
        self.attn_st, self.pad_st = 0, 1
        
        """Set ratio to 0.001 for low-resource setting and quickly debug."""
        self.ratio = 1
        
        """Preprocess the tokenized conversations."""
        self.data = []
        [gold_klg, rand_klg, swap_klg, gold_hist, rand_hist, swap_hist] = encode_data(tokenizer, data_path, data_cache)
        
        for idx, samp in enumerate(gold_klg[:int(self.ratio * len(gold_klg))]):
            self.data.append(samp)
            """Random pairing for 40% negetive samples; Entity swapping for 60% negetive samples."""
            if random.random() > neg_ratio:
                self.data.append(swap_klg[idx])
                pass
            else:
                self.data.append(rand_klg[idx])
            pass
        pass
        
    def instances(self, hist, reply, klg, label):        
        """knowledge form: [[BOS], gold knowledge]."""        
        inputs = hist + reply
        input_tti = [[self.hist] * (len(list(chain(*klg))))] + [[self.reply_st] * (len(list(chain(*reply))))]
        
        ins = {
            "input_ids": list(chain(*inputs)),
            "token_type_ids": list(chain(*input_tti)),
            "lm_label": label
        }

        """Make sure the sequences are equal length."""
        # print(len(ins["input_ids"]), len(ins["token_type_ids"])
        # set_trace()
        assert len(ins["input_ids"]) == len(ins["token_type_ids"])

        return ins
    
    def collate(self, batch):
        input_ids = pad_sequence([ins["input_ids"][0] for ins in batch], batch_first=self.batch_first, padding_value=self.pad)
        token_ids = pad_sequence([ins["token_type_ids"][0] for ins in batch], batch_first=self.batch_first, padding_value=self.pad_st)
        attn_ids = pad_sequence([ins["attention_mask"][0] for ins in batch], batch_first=self.batch_first, padding_value=self.attn_st)
        label_ids = [torch.tensor(ins["label"], dtype=torch.long) for ins in batch]

        return input_ids, token_ids, attn_ids, label_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


##############################################################################
# Info： Build training, validation and test data loaders.. 
##############################################################################
def build_loaders(args, tokenizer, logger):
    logger.info("Build training, validation and test data loaders")

    data_loaders = []
    
    if args.stage == "train":
        sets_type = ["train", "valid"]
        pass
    elif args.stage == "eval":
        sets_type = ["test"]
        pass
    else:
        raise Exception('Unknown dataset type to load.')
    
    for set_type in sets_type:
        dataset = NLIWikipediaDataset(tokenizer, seq_len=args.seq_len, data_path=os.path.join(args.data_path, set_type + '.json'), 
                                    data_cache=os.path.join(args.cache_path, set_type + '_cache'), neg_ratio=args.neg_ratio)
        
        print("Number of " + set_type + " samples: {:d}.".format(len(dataset)))
        
        """If args.distributed else None."""
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
        data_loaders.append(DataLoader(dataset,
                            collate_fn=dataset.collate,
                            pin_memory=(args.device == "cuda"),
                            num_workers=1,
                            sampler=data_sampler,
                            batch_size=args.batch_size if set_type=="train" else 1,
                            shuffle=False))
        pass

    return data_loaders

