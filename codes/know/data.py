import os
import json
import torch
from pdb import set_trace
from itertools import chain
from transformers import cached_path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


##############################################################################
# Info： Tokenize and encode the knowledge dataset. 
##############################################################################
def encode_data(tokenizer, data_path, data_cache):
    """Cache the processed knowledge data when first tokenizing for quick use."""
    data_cache = data_cache + "_" + type(tokenizer).__name__
    if data_cache and os.path.isfile(data_cache):
        print("Load tokenized knowledge dataset from cache at {:s}", data_cache)
        dataset = torch.load(data_cache)
        pass
    else:
        print("Process knowledge dataset from {:s}".format(data_path))
        plan_file = cached_path(data_path)
        with open(plan_file, "r", encoding="utf-8") as f:
            dataset = []
            samps = json.loads(f.read())
            for samp in samps:
                klg_enc = [tokenizer.encode(samp["knowledge"])]
                ent_enc = [tokenizer.encode(samp["masked_entity"])]
                new_data = {
                    "klg": klg_enc,
                    "ent": ent_enc
                }
                dataset.append(new_data)
                pass
            pass
        
        print("Tokenize and encode the knowledge dataset.")
        torch.save(dataset, data_cache)
        pass

    return dataset


##############################################################################
# Info： Generate a batch with three types.
##############################################################################
def get_klg_batch(batch_next, tokenizer, device="cuda"):
    input_ids, token_ids, label_ids = batch_next
    
    pos_ids = torch.arange(input_ids.size(1), dtype=torch.long)
    pos_ids = pos_ids.squeeze(0)
    
    """When evaluating, the sentence is inputted one-by-one in turn."""
    final_loss_mask = torch.ones(label_ids.size(1) - 1, dtype=torch.float)
    loss_mask = torch.masked_fill(final_loss_mask, torch.eq(label_ids[..., 1:].contiguous(), tokenizer.pad_token_id), 0.0)
    
    final_pos = torch.arange(label_ids.size(1), dtype=torch.long)
    pos_ids = final_pos.squeeze(0)
    
    """Put all tensor items to device."""
    input_ids = input_ids.to(torch.device(device))
    token_ids = token_ids.to(torch.device(device))
    label_ids = label_ids.to(torch.device(device))
    loss_mask = loss_mask.to(torch.device(device))
    pos_ids = pos_ids.to(torch.device(device))

    return  input_ids, token_ids, pos_ids, label_ids, loss_mask


##############################################################################
# Info： Make training data of "knowledge" for training process. 
##############################################################################
class KnowledgeDataset():
    def __init__(self, tokenizer, seq_len=512, data_path=None, data_cache=None, batch_first=True, is_label=True):
        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.batch_first = batch_first

        """Different separate token and token type ids."""
        self.klg_st, self.pad_st = 0, 1
        
        """Set ratio to 0.001 for low-resource setting and quickly debug."""
        self.ratio = 1
        
        """Preprocess the tokenized conversations."""
        self.samps = encode_data(tokenizer, data_path, data_cache)
        self.data = []

        for samp in self.samps[:int(self.ratio * len(self.samps))]:
            klg = samp["klg"]
            ent = samp["ent"]
            ins = self.instances(klg, ent)
        
            if len(ins["input_ids"]) <= seq_len:
                self.data.append(ins)
                pass
            else:
                continue
            pass
        pass
        
    def instances(self, klg, ent):                
        """knowledge form: [[BOS], gold knowledge]."""
        # set_trace()
        inputs = [[self.bos]] + klg + [[self.bos]] + ent + [[self.eos]]
        input_tti = [[self.klg_st] * (len(list(chain(*inputs))))]
        lm_label = [[self.pad] * (len(list(chain(*klg))) + 2)] + ent + [[self.eos]]
        
        ins = {
            "input_ids": list(chain(*inputs)),
            "token_type_ids": list(chain(*input_tti)),
            "lm_label": list(chain(*lm_label))
        }

        """Make sure the sequences are equal length."""
        # print(len(ins["input_ids"]), len(ins["token_type_ids"]), len(ins["lm_label"]))
        assert len(ins["input_ids"]) == len(ins["token_type_ids"]) == len(ins["lm_label"])

        return ins
    
    def collate(self, batch):
        input_ids = pad_sequence([torch.tensor(ins["input_ids"], dtype=torch.long) for ins in batch], 
                                 batch_first=self.batch_first, padding_value=self.pad)
        token_ids = pad_sequence([torch.tensor(ins["token_type_ids"], dtype=torch.long) for ins in batch], 
                                batch_first=self.batch_first, padding_value=self.pad_st)
        label_ids = pad_sequence([torch.tensor(ins["lm_label"], dtype=torch.long) for ins in batch], 
                                 batch_first=self.batch_first, padding_value=self.pad)

        return input_ids, token_ids, label_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


##############################################################################
# Info： Build knowledge loader.
##############################################################################
def klg_loader(args, tokenizer, logger):
    logger.info("Build knowledge loader")

    data_loaders = []
    for mode in ["train", "valid"]:
        dataset = KnowledgeDataset(tokenizer, seq_len=args.seq_len, data_path=os.path.join(args.data_path, mode + '.json'), 
                                    data_cache=os.path.join(args.cache_path, mode + '_cache'))
        
        print("Number of " + mode + " samples: {:d}".format(len(dataset)))
        
        """If args.distributed else None."""
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
        data_loaders.append(DataLoader(dataset,
                            collate_fn=dataset.collate,
                            pin_memory=(args.device == "cuda"),
                            num_workers=1,
                            sampler=data_sampler,
                            batch_size=args.batch_size if mode=="train" else 1,
                            shuffle=False))
        pass

    return data_loaders


