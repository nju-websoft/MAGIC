import os


from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import argparse
import torch
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer, TrainingArguments, Trainer, GPT2TokenizerFast
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from tqdm import tqdm
import json
from sklearn.cluster import KMeans
import random
import numpy as np
import datetime
from torch.utils.data import DataLoader
import sys


from generation_utils import KCenters
from generation_utils import check_and_create_folder
from generation_utils import compute_metrics
from generation_utils import remove_file_gene2
from model import AE


def combine_tensor(latent1, latent2):
    return torch.cat((latent1, latent2), dim=0)

def compute_ppl(model, sents, tokenizer = None):
    dataloader = DataLoader(sents, batch_size=1, shuffle=False, drop_last=False)
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    lst_ppl = []
    # tqdm_iter = tqdm(dataloader, desc='compute ppl')
    for data in dataloader:
        input = tokenizer(data, return_tensors='pt')
        input_ids = input['input_ids'].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
        lst_ppl.append(torch.exp(outputs[0]).item())
    avg_ppl = np.around(np.mean(lst_ppl), 2)
    return avg_ppl

parser = argparse.ArgumentParser()

parser.add_argument("--some_label", type=str, default='')
parser.add_argument("--model_prefix", type=str, default='')
parser.add_argument("--model_path", type=str, default='checkpoint-57000/pytorch_model.bin')
parser.add_argument("--Agnews_path", type=str, default="AG-data-7_3_sentiment.txt")
parser.add_argument("--property", nargs='+')


parser.add_argument("--output_dir", type=str, default="generated_txt")
parser.add_argument("--pretrained_encoder", type=str, default="bert-base-uncased")
parser.add_argument("--pretrained_decoder", type=str, default="gpt2-medium")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--latent_size", type=int, default=768)
parser.add_argument("--latent_num",type=int, default=1)
parser.add_argument("--seq_len_per_latent",type=int, default=20)

parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--pre_tokens",
    type=str,
    default=json.dumps(
        ['In summary','This essay discusses','Views on','The connection','Foundational to this is',
        'To review,','In brief,','An illustration of','Furthermore,','The central theme',
        'To conclude,','The key aspect','Prior to this','Emphasised are','To summarise',
        'The relationship','More importantly,','It has been shown','The issue focused on','In this essay',
        'Once upon a time','The book','The chicken','The city','The country',
        'The horse','The lake','The last time','The movie','The painting',
        'The pizza','The potato','The president of the country','The road','The year is 1910']
    )
)
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--variation", type=float, default=1e-3)

#Parameters for KCenters
parser.add_argument("--num_centers", type=int, default=1000)
parser.add_argument("--num_output_centers", type=int, default=10)
parser.add_argument("--topk", type=int, default=200)
parser.add_argument("--batch", type=int, default=5)
parser.add_argument("--max_iter", type=int, default=15)
parser.add_argument("--strategy", type=str, default='none', choices=('none', 'weight'))
parser.add_argument("--temperature", type=float, default=50)
parser.add_argument("--SDM_reinit", type=bool, default=True)
parser.add_argument("--weight",
    type=str, 
    default=json.dumps(
        [1,5,1]
    )
)
parser.add_argument("--config", type=str, default="generate_config_final.json")  

parser.add_argument("--file_loc", type=str, default="")
parser.add_argument("--log_res_root_path", type=str, default="probe_eval_log")

parser.add_argument(
    "--specify",
    type=str,
    default=None
)

args = parser.parse_args()



judgelist1 = args.property

print("!!! model used is {}".format(args.model_path))
_model_name = args.model_path.split("/")[-3] + "_" + args.model_path.split("/")[-2] 
print("!!! _model_name is {}".format(_model_name))


if isinstance(args.num_output_centers, int):
    args.num_output_centers = [[args.num_output_centers]*4]*2


encoder_tokenizer = BertTokenizer.from_pretrained(args.pretrained_encoder)
encoder = BertModel.from_pretrained(args.pretrained_encoder)
decoder_tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token


model = AE(encoder=encoder, decoder=decoder, args=args)
model.load_state_dict(torch.load(args.model_path), strict=False)
model.eval()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.no_cuda:
    device='cpu'
else:
    device='cuda'

model.to(device)


test_model_list = ['./model/Yelp2-checkpoint-64000', './model/AGnews-checkpoint-6000', './model/Toxic-checkpoint-3000']
tokenizer = DebertaV2Tokenizer.from_pretrained('deberta-v3-large')
loaded_model_list = []
for i in range(3):
    cur_model = DebertaV2ForSequenceClassification.from_pretrained(test_model_list[i], num_labels=2)
    loaded_model_list.append(cur_model)
topics = ["world","sports","business","science"]
task_list = ['sentiment', 'topic', 'detoxification']

test_args = TrainingArguments(
    output_dir='logs',
    do_train = False,
    do_predict = True,
    no_cuda = False,
    per_device_eval_batch_size=64,
    dataloader_drop_last = False,
    report_to='none'
)

base_model_ppl = "gpt2-large"
gpt2_ppl = GPT2LMHeadModel.from_pretrained(base_model_ppl).cuda()
tokenizer_ppl = GPT2TokenizerFast.from_pretrained('gpt2-large')


imdb_dataset = [{'sent':[]} for i in range(2)]
ag_dataset = [{'sent':[], 'imp_pos':[]} for i in range(4)]
toxic_dataset = [{'sent':[]} for i in range(2)]

with open('IMDb.txt', 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        imdb_dataset[label]['sent'].append(line[1].strip())

with open('Toxic.txt', 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        toxic_dataset[label]['sent'].append(line[1].strip())

with open(args.Agnews_path, 'r') as f:
    for line in f.readlines():
        line = json.loads(line)
        label = int(line[0])
        imp_label = int(line[1])
        ag_dataset[label]['sent'].append(line[-1].strip())
        ag_dataset[label]['imp_pos'].append(imp_label)

        label = int(line[0])    
        imp_label = int(line[1])
        ag_dataset[label]['sent'].append(line[-1].strip())
        ag_dataset[label]['imp_pos'].append(imp_label)
        
imdb_dataset = [Dataset.from_dict(i) for i in imdb_dataset]
ag_dataset = [Dataset.from_dict(i) for i in ag_dataset]
toxic_dataset = [Dataset.from_dict(i) for i in toxic_dataset]


imdb_dataloader = []
for dataset in imdb_dataset:
    tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
    imdb_dataloader.append(torch.utils.data.DataLoader(tmp_dataset, batch_size=32))

ag_dataloader = []
for dataset in ag_dataset:
    tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset = tmp_dataset.map(lambda e: {'pos_im_type': e['imp_pos']})
    tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'pos_im_type'])
    ag_dataloader.append(torch.utils.data.DataLoader(tmp_dataset, batch_size=64, shuffle=True))

toxic_dataloader = []
for dataset in toxic_dataset:
    tmp_dataset = dataset.map(lambda e: encoder_tokenizer(e['sent'], max_length=args.max_length, padding='max_length', truncation=True), batched=True)
    tmp_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
    toxic_dataloader.append(torch.utils.data.DataLoader(tmp_dataset, batch_size=32))


not_latents = None
sentiment_latents = {0:None, 1:None}
topic_latents = {0:None, 1:None, 2:None, 3:None}
topic_latents_flip = {0:None, 1:None, 2:None, 3:None}
topic_imp_label = {0:None, 1:None, 2:None, 3:None}

for i in range(2):
    for cnt in tqdm(iter(imdb_dataloader[i])):
        encoder_input_ids = cnt['input_ids']
        encoder_attention_mask = cnt['attention_mask']
        encoder_token_type_ids = cnt['token_type_ids']

        latent, encoder_output, past_key_values, tmp_latent1, tmp_latent2, flip_latent_z = model.encode(encoder_input_ids = encoder_input_ids, 
                                                               encoder_attention_mask = encoder_attention_mask, 
                                                               encoder_token_type_ids = encoder_token_type_ids,
                                                               pos_im_type = None,
                                                               is_adv = False)
        if sentiment_latents[i] is None:
            sentiment_latents[i] = latent.squeeze().detach()
        else:
            sentiment_latents[i] = torch.cat((sentiment_latents[i], latent.squeeze().detach()), dim=0)

for i in range(4):
    for cnt in tqdm(iter(ag_dataloader[i])):
        encoder_input_ids = cnt['input_ids']
        encoder_attention_mask = cnt['attention_mask']
        encoder_token_type_ids = cnt['token_type_ids']
        encoder_pos_label = cnt['pos_im_type']

        latent, encoder_output, past_key_values, tmp_latent1, tmp_latent2, flip_latent_z = model.encode(encoder_input_ids = encoder_input_ids, 
                                                               encoder_attention_mask = encoder_attention_mask, 
                                                               encoder_token_type_ids = encoder_token_type_ids,
                                                               pos_im_type = encoder_pos_label,
                                                               is_adv = True)
        if topic_latents[i] is None:
            topic_latents[i] = latent.squeeze().detach()
            topic_latents_flip[i] = flip_latent_z.squeeze().detach()
            topic_imp_label[i]  = encoder_pos_label
        else:
            topic_latents[i] = torch.cat((topic_latents[i], latent.squeeze().detach()), dim=0)
            topic_latents_flip[i] = torch.cat((topic_latents_flip[i], flip_latent_z.squeeze().detach()), dim=0)
            topic_imp_label[i] = torch.cat((topic_imp_label[i], encoder_pos_label), dim=0)

if args.model_prefix == "add_flip":
    for key, value in topic_latents.items():
        cur_id = key
        cur_tensor1 = value
        cur_tensor2 = topic_latents_flip[cur_id]
        cur_imp_label = topic_imp_label[cur_id]
        if judgelist1[-1] == "default":
            cur_senti = 0
        else:
            cur_senti = int(judgelist1[-1][0])
        adv_senti = abs(1-cur_senti)
        indices_adv  = torch.where(cur_imp_label == adv_senti)
        cur_tensor2 = cur_tensor2[indices_adv]
        topic_latents[cur_id] = combine_tensor(latent1 = cur_tensor1, latent2 = cur_tensor2)
else:
    print("No add_flip !!!!!!!!!!!!!!")

for cnt in tqdm(iter(toxic_dataloader[1])):
    encoder_input_ids = cnt['input_ids']
    encoder_attention_mask = cnt['attention_mask']
    encoder_token_type_ids = cnt['token_type_ids']
    
    latent, encoder_output, past_key_values, tmp_latent1, tmp_latent2, flip_latent_z = model.encode(encoder_input_ids = encoder_input_ids, 
                                                               encoder_attention_mask = encoder_attention_mask, 
                                                               encoder_token_type_ids = encoder_token_type_ids,
                                                               pos_im_type = None,
                                                               is_adv = False)
    if not_latents is None:
        not_latents = latent.squeeze().detach()
    else:
        not_latents = torch.cat((not_latents, latent.squeeze().detach()), dim=0)


current_time = datetime.datetime.now()
test_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
res_root_path = args.output_dir
res_path = os.path.join(res_root_path, "Probe_ckp-{}_t-{}".format(_model_name, test_time)) 

weight = json.loads(args.weight)
if args.config is not None:
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
        for keys in config:
            if keys == 'weight':
                weight = config['weight']
            if keys == 'num_output_centers':
                args.num_output_centers = config['num_output_centers']

kcmodel = KCenters(num_centers=args.num_centers, latent_size=args.latent_size, num_output_centers=args.num_output_centers, device='cuda')


judge_len1 = [1.5, 2.5, 3.5]
judge_len2 = [2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 10.5]


total_steps = len(judgelist1) * len(judge_len1) * len(judge_len2)
with tqdm(total=total_steps, desc="Training Progress") as pbar:  

    for out_name in judgelist1:
        num_index = out_name if out_name!="default" else "00"
        output_root = os.path.join(res_path, out_name) 
        check_and_create_folder(output_root)
        for j1 in judge_len1:
            for j2 in judge_len2:
                weight = json.loads(args.weight)
                if args.config is not None:
                    with open(args.config, 'r') as f:
                        config = json.loads(f.read())
                        for keys in config:
                            if keys == 'weight':
                                weight = config['weight']
                        
                weight[out_name][0] = j1
                weight[out_name][1] = j2
                if isinstance(weight, dict):
                    default_weight = weight['default']
                    weight_dict = [[default_weight for jt in range(4)]for it in range(2)]
                    for keys in weight:
                        if keys != 'default':
                            tmp_i = int(keys[0])
                            tmp_j = int(keys[1])
                            weight_dict[tmp_i][tmp_j] = weight[keys]
                else:
                    weight_dict = [[weight for jt in range(4)]for it in range(2)]

                output_text = []
                labels = []
                
                cur_output_dir = os.path.join(output_root, "predict_final_{}_{}.txt".format(j1, j2))
                
                cur_i = int(num_index[0])
                adv_i = abs(1-cur_i)
                cur_j = int(num_index[1])
                cur_sentiment_latents = sentiment_latents[cur_i].to('cuda')
                cur_topic_latents     = topic_latents[cur_j].to('cuda')
                weight = weight_dict[cur_i][cur_j]
                num_output_centers = args.num_output_centers[cur_i][cur_j]

                centers = kcmodel.multi_train(
                            [cur_sentiment_latents, cur_topic_latents, not_latents.to('cuda')],
                            weight=weight,
                            topk=args.topk,
                            SDM_reinit=args.SDM_reinit,
                            max_iter=args.max_iter,
                            strategy=args.strategy,
                            temperature=args.temperature,
                            num_output_centers=num_output_centers
                            ).cpu().numpy()

                centers = [torch.FloatTensor(k).unsqueeze(0) for k in centers]     

                for prompts in json.loads(args.pre_tokens):
                    tokens = decoder_tokenizer(prompts, return_tensors='pt')
                    input_ids = tokens.input_ids
                    attention_mask = tokens.attention_mask
                    input_ids = input_ids.expand(args.batch_size, -1)
                    attention_mask = attention_mask.expand(args.batch_size, -1)

                    output = model.generate(
                        input_latent=random.choice(centers),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        variation=args.variation,
                        max_len=50,
                        rp=1.2
                    )

                    output_text.extend(decoder_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True))
                    labels.extend([[cur_i,cur_j,1]] * args.batch_size)
                    assert len(labels) == len(output_text)

                pbar.update(1)
                
                with open(cur_output_dir, 'w') as f:
                    for i in range(len(output_text)):
                        f.write(json.dumps([labels[i], output_text[i]])+'\n')

assert len(judgelist1) == 1
cur_para = judgelist1[0]

if cur_para == "default":
    args.specify = "[0,0,1]"
else:
    args.specify = "[{},{},1]".format(cur_para[0], cur_para[1])


inner_loop = judge_len1
outer_loop = judge_len2                    


eval_res_path = os.path.join(args.log_res_root_path,  "alldata_Ana_Probe_ckp-{}_{}_t-{}".format(_model_name, args.model_prefix, test_time)) 
out_root_path = os.path.join(eval_res_path, cur_para) 
check_and_create_folder(out_root_path)
output_path = os.path.join(out_root_path, "result.txt") 
outpu_file = open(output_path, 'a')
outpu_file.writelines(f"{args.some_label}\n")
outpu_file.writelines(f"path of txt file read is {os.path.join(res_path, cur_para)}\n")
outpu_file.flush()

for inner in inner_loop:
    for outer in outer_loop:
  
        genTxt_output_root = os.path.join(res_path, cur_para)
    
        args.file_loc = os.path.join(genTxt_output_root, "predict_final_{}_{}.txt".format(str(inner), str(outer)))
    
        ppl_sent = []
        dataset = {'label':[], 'sent':[]}
        with open(args.file_loc, 'r') as f:
            for line in f.readlines():
                label, sent = json.loads(line.strip())
                dataset['label'].append(label)
                dataset['sent'].append(sent.strip())
                ppl_sent.append(sent.strip())

        dataset = Dataset.from_dict(dataset)
        train_out = {}

        if args.specify is not None:
            specify = json.loads(args.specify)
            dataset = dataset.filter(lambda e: e['label'] == specify)

        for i in range(3):
            if args.specify is not None:
                if specify[i] == -1:
                    continue
            model = loaded_model_list[i]
            eval_dataset = None
            if 'AGnews' in test_model_list[i]:
                eval_dataset = dataset.map(lambda e: tokenizer(topics[e['label'][i]]+'[SEP]'+e['sent'], truncation=True, padding='max_length', max_length=100))
                eval_dataset = eval_dataset.map(lambda e: {'labels': 1})
            else:
                eval_dataset = dataset.map(lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=100), batched=True)
                eval_dataset = eval_dataset.map(lambda e: {'labels': e['label'][i]})
            eval_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

            trainer = Trainer(
                model=model,
                args = test_args,
                compute_metrics=compute_metrics, 
            )
            train_out[task_list[i]] = trainer.evaluate(eval_dataset)['eval_accuracy']

        current_ppl = compute_ppl(gpt2_ppl, ppl_sent, tokenizer = tokenizer_ppl)

        out_str = ""
        for key, value in train_out.items():
            out_str += "{}: {},  ".format(key, value) 
        out_str += f"ppl: {current_ppl}"
        outpu_file.writelines("[{},{},1]".format(inner, outer) + '\n')
        outpu_file.writelines(out_str + '\n')
        outpu_file.flush()


outpu_file.close()
            