import torch

from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
import random
import numpy as np
import json
from tqdm import tqdm
import argparse
import shutil
import os
PRETRAINED_ENCODER = "/data1/yiliu/Pre_Train_Model/bert-base-uncased"
PRETRAINED_DECODER = "/data1/yiliu/Pre_Train_Model/gpt2-medium"


def create_file(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{folder_path}': {str(e)}")
    else:
        print(f"Folder '{folder_path}' already exists.")
    



def optim(centers, matrix, topk, batch=100, latent_size = 768):
    tmp_score = - distance(centers, matrix)   # 5, 5000
    tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1)
    
    tot_num = tmp_indices.shape[0]
    epoch = tot_num//batch + (1 if tot_num % batch != 0 else 0)

    new_centers = None
    for i in range(epoch):
        start = i * batch
        end = i * batch + batch
        if end > tot_num:
            end = tot_num
        
        tmp_centers = torch.mean(torch.gather(matrix.unsqueeze(0).expand(end-start,-1,-1), 1, tmp_indices[start:end].unsqueeze(-1).expand(-1,-1,latent_size)),dim=1).squeeze()
         
        if new_centers is None:
            new_centers = tmp_centers
        else:
            new_centers = torch.cat([new_centers, tmp_centers], dim=0)
    return  new_centers


def optim_re(centers, matrix, topk, batch=100, latent_size = 768, latent_0 = None, latent_1 = None, tmp_z = None):
    part1 = 0.8
    part2 = 0.2
    latent_0_mean = latent_0.mean(dim=0, keepdim=True) # 1, 768
    latent_1_mean = latent_1.mean(dim=0, keepdim=True) # 1, 768
    latent_1_mean_all = latent_1_mean.expand(tmp_z.shape[0], -1)
    new_matrix =  part1 * tmp_z + part2 *  latent_1_mean_all
    tmp_score = - distance(centers, matrix)   # 5, 5000
    tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1)
    
    tot_num = tmp_indices.shape[0]
    epoch = tot_num//batch + (1 if tot_num % batch != 0 else 0)

    new_centers = None
    for i in range(epoch):
        start = i * batch
        end = i * batch + batch
        if end > tot_num:
            end = tot_num
        
        tmp_centers = torch.mean(torch.gather(new_matrix.unsqueeze(0).expand(end-start,-1,-1), 1, tmp_indices[start:end].unsqueeze(-1).expand(-1,-1,latent_size)),dim=1).squeeze()
         
        if new_centers is None:
            new_centers = tmp_centers
        else:
            new_centers = torch.cat([new_centers, tmp_centers], dim=0)
    return  new_centers

def distance(matrix1, matrix2, batch=100):
        '''
        Input:
            matrix1: FloatTensor(i * m)
            matrix2: FloatTensor(j * m)
        Output:
            distance matrix: FloatTensor(i * j)
        '''

        assert len(matrix1.shape) == 2
        assert len(matrix2.shape) == 2

        dis = None
        tot_num = matrix1.shape[0]
        epoch = tot_num//batch + (1 if tot_num % batch != 0 else 0)
        matrix1 = matrix1.unsqueeze(dim=1)  # 5 1 768
        matrix2 = matrix2.unsqueeze(dim=0)  # 1 5000 768
        for i in range(epoch):
            start = i * batch
            end = i * batch + batch
            if end > tot_num:
                end = tot_num
            tmp_matrix1 = matrix1[start:end]
            tmp_dis = (tmp_matrix1 - matrix2) ** 2.0  # 5 5000 768
            tmp_dis = torch.sum(tmp_dis, dim=-1).squeeze() # 5 5000
            if dis is None:
                dis = tmp_dis
            else:
                dis = torch.cat([dis, tmp_dis], dim=0)

        
        return dis

