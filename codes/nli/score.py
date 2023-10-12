import os
import json
import torch
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
            # set_trace()
            dataset = []
            for data in f.readlines():
                """Sentence-pairing: knowledge as prompt, response as the next_sentence."""
                data = json.loads(data)
                sent_pair = tokenizer(data["knowledge"], data["response"].replace("<|endoftext|>", ""), return_tensors='pt')                
                dataset.append(sent_pair)
                pass
            pass
        
        print("Tokenize and encode the dataset.")
        torch.save(dataset, data_cache)
        pass

    return dataset


##############################################################################
# Info： Make training data of dialogues containing "context", "response" and 
#        "knowledge" for training process. 
##############################################################################
class GENWikipediaDataset():
    def __init__(self, tokenizer, testset=None, task="fact", batch_first=True, is_label=True):
        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.is_label = is_label
        self.task = task

        """Different separate token and token type ids."""
        self.attn_st, self.pad_st = 0, 1

        """Set ratio to 0.001 for low-resource setting and quickly debug."""
        self.ratio = 1
        self.data = []
        
        self.data.extend(testset[:int(self.ratio * len(testset))])

    def collate(self, batch):
        input_ids = pad_sequence([ins["input_ids"][0] for ins in batch], 
                                 batch_first=self.batch_first, padding_value=self.pad)
        token_ids = pad_sequence([ins["token_type_ids"][0] for ins in batch], 
                                 batch_first=self.batch_first, padding_value=self.pad_st)
        attn_ids = pad_sequence([ins["attention_mask"][0] for ins in batch], 
                                batch_first=self.batch_first, padding_value=self.attn_st)

        return input_ids, token_ids, attn_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


##############################################################################
# Info： Build training, validation and test data loaders.. 
##############################################################################
def infer_loaders(args, tokenizer, logger):
    logger.info("Build inference data loaders.")
    
    testset = encode_data(tokenizer, data_path=os.path.join(args.data_path, 'test.json'), 
                data_cache=os.path.join(args.cache_path, 'test'))
    
    dataset = GENWikipediaDataset(tokenizer, testset=testset)
    
    print("Number of test samples for inference: {:d}".format(len(dataset)))
    
    """If args.distributed else None."""
    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    data_loader = DataLoader(dataset,
                    collate_fn=dataset.collate,
                    pin_memory=(args.device == "cuda"),
                    num_workers=1,
                    sampler=data_sampler,
                    batch_size=args.batch_size,
                    shuffle=False)
        
    return [data_loader]

