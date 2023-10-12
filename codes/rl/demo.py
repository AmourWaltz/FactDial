# imports
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch
# from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
# from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt, listify_batch

import random
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification , AutoTokenizer, pipeline

# get models
gen_model = AutoModelForCausalLMWithValueHead.from_pretrained('/home/ma-user/work/byxue/models/gpt2-dialogbot-base-chinese')
model_ref = create_reference_model(gen_model)
tokenizerOne = AutoTokenizer.from_pretrained('/home/ma-user/work/byxue/models/gpt2-dialogbot-base-chinese/', padding_side='left')
# tokenizerOne.eos_token_id = tokenizerOne.sep_token_id

ts_texts = ["我喜欢下雨。", "我讨厌他."]
cls_model = AutoModelForSequenceClassification.from_pretrained("/home/ma-user/work/byxue/models/c2-roberta-base-finetuned-dianping-chinese", num_labels=2)
tokenizerTwo = AutoTokenizer.from_pretrained("/home/ma-user/work/byxue/models/c2-roberta-base-finetuned-dianping-chinese")
classifier = pipeline('sentiment-analysis', model=cls_model, tokenizer=tokenizerTwo)
classifier(ts_texts)


from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import json

data = []
with open("/home/ma-user/work/byxue/data/luge_Diamante/train.txt", "r", encoding="utf-8") as f:
    for i in f.readlines():
        line = json.loads(i)
        data.append(line)


def preprocess_conversation(data):
    sep_id = tokenizerOne.sep_token_id
    cls_id = tokenizerOne.cls_token_id
    dialogue_list = []
    for conver in data:
        input_ids = [cls_id]
        start = conver["conversation"][0]
        # print(start["utterance"])
        input_ids += tokenizerOne.encode(start["utterance"], add_special_tokens=False)
        if len(input_ids) > 7:
            input_ids.append(sep_id)
            dialogue_list.append(input_ids)
        else:
            continue

    return dialogue_list


dialogue_list = preprocess_conversation(data)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        return torch.tensor(x)

    def __len__(self):
        return len(self.data)
    
mydataset = MyDataset(dialogue_list)

def collate_fn(batch):
    padded_batch = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=tokenizerOne.sep_token_id)
    return padded_batch

config = PPOConfig(
    model_name="gpt2-positive",
    learning_rate=1.41e-5,
    steps = 2000,
    batch_size = 16
)


ppo_trainer = PPOTrainer(config, gen_model, model_ref, tokenizerOne, dataset=mydataset, data_collator=collate_fn)

rewards_list = []
for epoch, batch in enumerate(ppo_trainer.dataloader):
    #### Get response from gpt2
    query_tensors = []
    response_tensors = []
    query_tensors = [torch.tensor(t).long() for t in batch]
    for query in batch:
        input_ids = query.unsqueeze(0)
        response = []
        for _ in range(30):
            outputs = ppo_trainer.model(input_ids=input_ids)
            logits = outputs[0]
            next_token_logits = logits[0, -1, :]
            next_token_logits[ppo_trainer.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            if next_token == ppo_trainer.tokenizer.sep_token_id:  #
                break
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            response.append(next_token.item())
        response_tensors.append(torch.Tensor(response).long())
    responseSet = ["".join(ppo_trainer.tokenizer.convert_ids_to_tokens([i.item() for i in r])) for r in response_tensors]
    print(responseSet)
    #### Get reward from sentiment model
    pipe_outputs = classifier(responseSet)
    rewards = [torch.tensor(output["score"]) for output in pipe_outputs]

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    print("epoch{}, reword is {}".format(epoch, sum(rewards)))
    rewards_list.append(sum(rewards))