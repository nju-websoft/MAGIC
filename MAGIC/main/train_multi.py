import os
import argparse
import torch
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import json
import wandb
from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict, Sequence
import random
from utils import create_file
from model import AE


parser = argparse.ArgumentParser()


parser.add_argument("--pretrained_encoder", type=str, default="bert-base-uncased")
parser.add_argument("--pretrained_decoder", type=str, default="gpt2-medium")

parser.add_argument('--model_dir', default='')
parser.add_argument('--Ag_News_Path', default='AG-data-7_3_sentiment.txt')
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--latent_size", type=int, default=768)
parser.add_argument("--latent_num",type=int, default=1)
parser.add_argument("--seq_len_per_latent",type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epoch",type=int, default=300)
parser.add_argument("--lr",type=float, default=1e-4)
parser.add_argument("--fp16", default = True)
parser.add_argument("--wandb", default = True)
parser.add_argument("--no_fix", action="store_true")
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--contrasitive_loss", type=float, default=None)
parser.add_argument("--sparse_loss", type=float, default=0.4)
parser.add_argument("--latent_classify_loss", type=float, default=0.2)
parser.add_argument("--aspect_gap_loss", type=float, default=0.3)  
parser.add_argument("--variation", type=float, default=1e-3)
parser.add_argument("--classifier_head_num", type=int, default=3)
parser.add_argument("--classifier_class_num_per_head", type=str, default='[2,2,4]')
parser.add_argument("--classifier_mid_size", type=int, default=128)
parser.add_argument("--classifier_head_type", type=str, default='multiple', choices=('single', 'multiple'))
parser.add_argument("--aspect_gap_head_num", type=int, default=3)
parser.add_argument("--aspect_gap_amplification", type=int, default=10)
args = parser.parse_args()



if args.wandb:
    wandb.login()
    wandb.init(project="CTG-multi", entity="", name="")

encoder_tokenizer = BertTokenizer.from_pretrained(args.pretrained_encoder)
encoder = BertModel.from_pretrained(args.pretrained_encoder)
decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

model = AE(encoder=encoder, decoder=decoder, args=args)

loss_list = {}
latent_classify_args = None
aspect_gap_args = None

if args.latent_classify_loss is not None:
    loss_list['latent_classify_loss'] = args.latent_classify_loss
    latent_classify_args = {
        'head_num':args.classifier_head_num,
        'class_num_per_head':[2,2,4],       
        'mid_size':args.classifier_mid_size,
        'head_type':args.classifier_head_type
    }
if args.aspect_gap_loss is not None:
    loss_list['aspect_gap_loss'] = args.aspect_gap_loss
    aspect_gap_args = {
        'head_num':args.aspect_gap_head_num,
        'amplification':args.aspect_gap_amplification
    }

if args.sparse_loss is not None:
    loss_list["sparse_loss"] = args.sparse_loss


if len(loss_list) == 0:
    loss_list = None

model.set_losslist(loss_list, latent_classify_args, aspect_gap_args)

if not args.no_fix:
    model.fix_decoder()

dataset = [{'sent':[], 'type':[]} for i in range(2)]
dataset.append({'sent':[], 'type':[], 'im_type':[]})


with open(args.IMDB_Path, 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        dataset[0]['sent'].append(line[1].strip())
        dataset[0]['type'].append(int(line[0]))

with open(args.Toxic_Path, 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        dataset[1]['sent'].append(line[1].strip())
        dataset[1]['type'].append(int(line[0]))

with open(args.Ag_News_Path, 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        dataset[2]['sent'].append(line[2].strip())
        dataset[2]['type'].append(int(line[0]))
        dataset[2]['im_type'].append(int(line[1]))



columns = ['encoder_input_ids', 'encoder_attention_mask', 'encoder_token_type_ids', 'type', 'decoder_input_ids', 'decoder_attention_mask'] 

if 'latent_classify_loss' in loss_list:
    columns.extend(['pos_label', 'neg_labels'])
columns.extend(['pos_im_type'])

train_dataset = {i:[] for i in columns}
train_dataset['head_index']=[]

for i in range(3):
    tmp_dataset = Dataset.from_dict(dataset[i])
    tmp_dataset = tmp_dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset = tmp_dataset.rename_columns({'input_ids':'encoder_input_ids', 'attention_mask':'encoder_attention_mask', 'token_type_ids':'encoder_token_type_ids'})
    tmp_dataset = tmp_dataset.map(lambda e: decoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset = tmp_dataset.rename_columns({'input_ids':'decoder_input_ids', 'attention_mask':'decoder_attention_mask'})
    if 'contrasitive_loss' in loss_list:
        tmp_dataset = tmp_dataset.map(lambda e: encoder_tokenizer(e['adv_sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
        tmp_dataset = tmp_dataset.rename_columns({'input_ids':'adv_input_ids', 'attention_mask':'adv_attention_mask', 'token_type_ids':'adv_token_type_ids'})
    if 'latent_classify_loss' in loss_list:
        tmp_dataset = tmp_dataset.map(lambda e: {'pos_label': e['type'], 'neg_labels': [1 - e['type']]})
    
    if i == 2:
        tmp_dataset = tmp_dataset.map(lambda e: {'pos_im_type': e['im_type']})
    else:
        tmp_dataset = tmp_dataset.map(lambda e: {'pos_im_type': e['type'] })

    tmp_dataset.set_format(type='torch', columns=columns)

    tmp_dataloader = torch.utils.data.DataLoader(tmp_dataset, batch_size=args.batch_size, shuffle=True)
    for cnt in iter(tmp_dataloader):   
        for k in columns:
            train_dataset[k].append(cnt[k])
        train_dataset['head_index'].append(i)
train_dataset = Dataset.from_dict(train_dataset)
train_dataset.set_format(columns=columns+['head_index'])



list_result_cra = [[],[],[],[],[],[],[],[]]
for _i in tqdm(range(len(dataset[2]["sent"]))):
    cur_sent    = dataset[2]["sent"][_i]
    cur_type    = dataset[2]["type"][_i]
    cur_im_type = dataset[2]["im_type"][_i]
    cur_res = encoder_tokenizer(cur_sent, max_length=100, padding='max_length', truncation=True)
    cur_input_ids      = cur_res["input_ids"]
    cur_attention_mask = cur_res["attention_mask"]
    cur_token_type_ids = cur_res["token_type_ids"]
    cur_all = [cur_input_ids, cur_attention_mask, cur_token_type_ids]
    cur_all = np.array(cur_all)                      
    indexs = cur_type * 2 + abs(cur_im_type-1)   
    list_result_cra[indexs].append(cur_all)

for _i in range(len(list_result_cra)):                 
    list_result_cra[_i] = np.array(list_result_cra[_i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    list_adv_all: None

    def __call__(self, instances: Sequence[Dict]):
        
        encoder_input_ids, encoder_attention_mask, encoder_token_type_ids, decoder_input_ids, decoder_attention_mask, pos_label, neg_labels, head_index, pos_im_type= tuple([instance[key] for instance in instances] for key in ("encoder_input_ids", "encoder_attention_mask", "encoder_token_type_ids", "decoder_input_ids", "decoder_attention_mask", "neg_labels", "head_index", "pos_im_type"))

        adv_encoder_input_ids = None
        adv_encoder_attention_mask = None
        adv_encoder_token_type_ids = None
        if head_index[-1] == 2:   
            for _i in range(len(pos_label[0])):
                type_topic = pos_label[0][_i]
                type_imp_att = pos_im_type[0][_i]
                cur_index = 2*type_topic + type_imp_att


                cur_index_inner_list = random.sample(range(0, len(self.list_adv_all[cur_index])), 4)

                if adv_encoder_input_ids is None:

                    adv_encoder_input_ids      = torch.tensor(self.list_adv_all[cur_index][cur_index_inner_list][:,0]).unsqueeze(0) 
                    adv_encoder_attention_mask = torch.tensor(self.list_adv_all[cur_index][cur_index_inner_list][:,1]).unsqueeze(0)
                    adv_encoder_token_type_ids = torch.tensor(self.list_adv_all[cur_index][cur_index_inner_list][:,2]).unsqueeze(0)

                else:
                    adv_encoder_input_ids      = torch.cat((adv_encoder_input_ids,      torch.tensor(self.list_adv_all[cur_index][cur_index_inner_list][:,0]).unsqueeze(0)), dim=0)
                    adv_encoder_attention_mask = torch.cat((adv_encoder_attention_mask, torch.tensor(self.list_adv_all[cur_index][cur_index_inner_list][:,1]).unsqueeze(0)), dim=0)
                    adv_encoder_token_type_ids = torch.cat((adv_encoder_token_type_ids, torch.tensor(self.list_adv_all[cur_index][cur_index_inner_list][:,2]).unsqueeze(0)), dim=0)

        encoder_input_ids = torch.tensor(encoder_input_ids)
        encoder_attention_mask = torch.tensor(encoder_attention_mask)
        encoder_token_type_ids = torch.tensor(encoder_token_type_ids)
        decoder_input_ids = torch.tensor(decoder_input_ids)
        decoder_attention_mask = torch.tensor(decoder_attention_mask)
        pos_label = torch.tensor(pos_label)
        neg_labels = torch.tensor(neg_labels)
        head_index = torch.tensor(head_index)
        pos_im_type = torch.tensor(pos_im_type)


        return dict(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_token_type_ids=encoder_token_type_ids,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            pos_label = pos_label,
            neg_labels = neg_labels,
            head_index = head_index,
            pos_im_type = pos_im_type,
            adv_encoder_input_ids = adv_encoder_input_ids,
            adv_encoder_attention_mask = adv_encoder_attention_mask,
            adv_encoder_token_type_ids = adv_encoder_token_type_ids
        )



training_args = TrainingArguments(
    output_dir=args.model_dir,
    learning_rate=args.lr,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=1,
    logging_dir='./logs',
    logging_steps=100,
    do_train=True,
    do_eval=False,
    no_cuda=args.no_cuda,
    save_strategy="steps",
    save_steps=1000,
    fp16=args.fp16,
    report_to='wandb' if args.wandb else 'none'
)



create_file(args.model_dir)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSupervisedDataset(list_adv_all=list_result_cra)
)
train_out = trainer.train()