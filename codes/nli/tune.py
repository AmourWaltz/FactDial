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
            verif_set, hallu_set, fact_set = [], [], []
            data_pool = json.loads(f.read())
            
            for idx, data in enumerate(data_pool):
                if idx > 0:
                    """Sentence-pairing: knowledge as prompt, response as the next_sentence."""
                    sent_pair = tokenizer(data["klg"], data["reply"], return_tensors='pt')
                    sent_pair["label"] = 1 if data["verif"] == True else 0
                    verif_set.append(sent_pair)
                    
                    if data["verif"] == True:
                        sent_pair["label"] = 1 if data["hallu"] == True else 0
                        hallu_set.append(sent_pair)
                    
                        sent_pair["label"] = data["fact_acc"]
                        fact_set.append(sent_pair)
                        pass
                    pass
                pass
            pass
        
        dataset = [verif_set, hallu_set, fact_set]
        print("Tokenize and encode the dataset.")
        torch.save(dataset, data_cache)
        pass

    return dataset


##############################################################################
# Info： Make training data of dialogues containing "context", "response" and 
#        "knowledge" for training process. 
##############################################################################
class SNLIWikipediaDataset():
    def __init__(self, tokenizer, datasets=None, mode="train", task="fact", batch_first=True, is_label=True):
        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.batch_first = batch_first
        self.is_label = is_label
        self.task = task

        """Different separate token and token type ids."""
        self.attn_st, self.pad_st = 0, 1

        """Set ratio to 0.001 for low-resource setting and quickly debug."""
        self.ratio = 1
        self.data = []
        # set_trace()
        [verif_set, hallu_set, fact_set] = datasets
        
        if task == "verif":
            dataset = verif_set
            pass
        elif task == "hallu":
            dataset = hallu_set
            pass
        elif task == "fact":
            dataset = fact_set
            pass
        
        if mode == "train":
            self.data.extend(dataset[:int(self.ratio * len(dataset))])
            pass
        elif mode == "valid":
            self.data.extend(dataset[:int(self.ratio * 0.1 * len(dataset))])
            pass
        elif mode == "test":
            self.data.extend(dataset[int(self.ratio * 0.9 * len(dataset)):])
            pass
        pass

    def collate(self, batch):
        input_ids = pad_sequence([ins["input_ids"][0] for ins in batch], 
                                 batch_first=self.batch_first, padding_value=self.pad)
        token_ids = pad_sequence([ins["token_type_ids"][0] for ins in batch], 
                                 batch_first=self.batch_first, padding_value=self.pad_st)
        attn_ids = pad_sequence([ins["attention_mask"][0] for ins in batch], 
                                batch_first=self.batch_first, padding_value=self.attn_st)
        if self.task == "fact":
            label_ids = torch.stack([torch.tensor(ins["label"], dtype=torch.float) for ins in batch])
        else:
            label_ids = torch.stack([torch.tensor(ins["label"], dtype=torch.long) for ins in batch])

        return input_ids, token_ids, attn_ids, label_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


##############################################################################
# Info： Build training, validation and test data loaders.. 
##############################################################################
def tune_loaders(args, tokenizer, logger):
    logger.info("Build training, validation and test data loaders")
    
    if args.stage == "train":
        sets_type = ["train", "valid"]
        pass
    elif args.stage == "tune":
        sets_type = ["test"]
        pass
    else:
        raise Exception('Unknown dataset type to load.')
    
    datasets = encode_data(tokenizer, data_path=os.path.join(args.data_path, 'conv_fever.json'), 
                data_cache=os.path.join(args.cache_path, 'conv_fever_cache'))
    data_loaders = []
    
    for set_type in sets_type:
        dataset = SNLIWikipediaDataset(tokenizer, mode=set_type, datasets=datasets, task=args.task)
        
        print("Number of " + set_type + " samples: {:d}".format(len(dataset)))
        
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

