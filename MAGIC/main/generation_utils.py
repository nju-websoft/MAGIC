import torch
import os
from tqdm import tqdm
import random
from sklearn.cluster import KMeans
import shutil

def remove_file_gene(destination_dir, main_file = "probe.py"):
    list1 = [main_file, "generation_utils.py"]
    source_dir = "main/"
    for file_name in list1:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        try:
            shutil.copy(source_path, destination_path)
            print(f"Copied  {file_name} to {destination_dir}")
        except FileNotFoundError:
            print(f"File {file_name} not found in {source_dir}")
        except Exception as e:
            print(f"Error copying {file_name}: {str(e)}")

def remove_file_gene2(destination_dir, source_dir = "/main/", main_file = "probe.py"):
    list1 = [main_file]
    for file_name in list1:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        try:
            shutil.copy(source_path, destination_path)
            print(f"Copied  {file_name} to {destination_dir}")
        except FileNotFoundError:
            print(f"File {file_name} not found in {source_dir}")
        except Exception as e:
            print(f"Error copying {file_name}: {str(e)}")

def check_and_create_folder(path):
    if os.path.exists(path):
        print("exist") 
    else:
        os.makedirs(path)
        print("created")

def compute_metrics(pred):
    labels = torch.tensor(pred.label_ids).long()
    preds = torch.softmax(torch.tensor(pred.predictions),dim=-1)
    probs = torch.gather(preds, 1,labels.view(-1, 1))
    acc = torch.mean(probs).item()
    
    return {
        'accuracy': acc,
    }

class KCenters:
    def __init__(
        self,
        num_centers,
        latent_size,
        num_output_centers,
        device,
    ):
        self.num_centers = num_centers
        self.num_output_centers = num_output_centers
        self.device = device
        self.latent_size = latent_size
        self.centers = None
        self.score = None


    def init_cluster_center(self, num_centers, latent_size):
        '''
        '''
        if num_centers == 0:
            clusters = None
        else:
            clusters = torch.rand(num_centers, latent_size) * 2 - 1
        return clusters


    def Sparse_Distributed_Memory_Reinitalization(self, X, topk):
        length = len(X)
        self.centers = None
        for i in range(length):
            query_matrix = X[i]
            query_centers = torch.zeros_like(query_matrix).to(self.device)
            for j in range(length):
                if j !=i :
                    key_matrix = X[j]
                    query_centers += self.optim(query_matrix, key_matrix, topk, 'none')
            query_centers = (query_centers + query_matrix) / length

            query_score = torch.zeros(query_centers.shape[0]).to(self.device)
            for matrix in X:
                tmp_score = -self.distance(query_centers, matrix)
                tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1)
                query_score += torch.mean(tmp_values, dim=-1)
            query_score = query_score/length
            query_values, query_indices = torch.topk(query_score, k=self.num_centers)
            query_centers = torch.index_select(query_centers, 0, query_indices)

            if self.centers is None:
                self.centers = query_centers
            else:
                self.centers = torch.cat([self.centers, query_centers], dim=0)
        

        scores = torch.zeros(self.centers.shape[0]).to(self.device)
        for matrix in X:
            tmp_score = -self.distance(self.centers, matrix)
            tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1)
            scores += torch.mean(tmp_values, dim=-1)
        scores = scores/length
        out_values, out_indices = torch.topk(scores, k=self.num_centers)
        self.centers = torch.index_select(self.centers, 0, out_indices)
        


    def Kernel_Density_Estimation(self):
        return



    def train(
            self,
            X,
            weight=[1,1,1],
            topk=50,
            max_iter=1,
            strategy='none',
            SDM_reinit=False,
            tol=1e-10,
            temperature=50,
            num_output_centers=None,
            flip_z = None,
            imp_lable = None,
            adv_i = None
        ):
        

        assert strategy in {'none', 'weight'}
        length = sum(weight)

        if num_output_centers is not None:
            self.num_output_centers = num_output_centers

        if SDM_reinit:
            self.Sparse_Distributed_Memory_Reinitalization(X, topk)

        if strategy in {'none', 'weight'}:

            for i in range(max_iter):
            # for i in range(2):
                new_centers = torch.zeros_like(self.centers).to(self.device)
                
                for j in range(len(X)):
                    matrix = X[j]
                    w = weight[j]
                    cur_flip = None
                    cur_lable = None
                    if j == 1:
                        cur_flip = flip_z
                        cur_lable = imp_lable
                    new_centers += w * self.optim(self.centers, matrix, topk, strategy, temperature=temperature, flip_z=cur_flip, imp_label=cur_lable, adv_i= adv_i)
                new_centers = new_centers/length
                

                self.centers = new_centers
                        
                        

            
        
        new_score = torch.zeros(self.centers.shape[0]).to(self.device)
        for matrix in X:
            tmp_score = -self.distance(self.centers, matrix)
            tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1)
            new_score += torch.mean(tmp_values, dim=-1)
        self.score = new_score/length


        out_values, out_indices = torch.topk(self.score, k=self.num_output_centers)
        return torch.index_select(self.centers, 0, out_indices)

    
    def multi_train(
            self,
            X,
            weight=[1,1,1],
            topk=50,
            max_iter=1,
            strategy='none',
            SDM_reinit=False,
            tol=1e-10,
            temperature=50,
            num_output_centers=None,
            flip_z = None,
            imp_lable = None,
            adv_i = None
        ):

        assert strategy in {'none', 'weight'}
        length = sum(weight)

        if num_output_centers is not None:
            self.num_output_centers = num_output_centers

        if SDM_reinit:
            self.Sparse_Distributed_Memory_Reinitalization(X, topk)

        if strategy in {'none', 'weight'}:

            # for i in range(max_iter):
            for i in range(15):
                new_centers = torch.zeros_like(self.centers).to(self.device)
                
                for j in range(len(X)):
                    matrix = X[j]
                    w = weight[j]
                    cur_flip = None
                    cur_lable = None
                    if j == 1:
                        cur_flip = flip_z
                        cur_lable = imp_lable
                    new_centers += w * self.optim(self.centers, matrix, topk, strategy, temperature=temperature, flip_z=cur_flip, imp_label=cur_lable, adv_i= adv_i)
                new_centers = new_centers/length
                

                self.centers = new_centers
                        
        
        new_score = torch.zeros(self.centers.shape[0]).to(self.device)
        for matrix in X:
            tmp_score = -self.distance(self.centers, matrix)
            tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1)
            new_score += torch.mean(tmp_values, dim=-1)
        self.score = new_score/length


        out_values, out_indices = torch.topk(self.score, k=self.num_output_centers)
        return torch.index_select(self.centers, 0, out_indices)

   
    def optim(self, centers, matrix, topk, strategy, batch=100, temperature=50, flip_z = None, imp_label = None, adv_i = None):
        tmp_score = - self.distance(centers, matrix)
        tmp_values, tmp_indices = torch.topk(tmp_score, k=topk, dim=-1) 
        
        if imp_label is not None:

            mean_value = list() 
            adv_tmp_indices = torch.zeros_like(tmp_indices).cuda()
            indices_adv  = torch.where(imp_label == adv_i)[0]
            for ti in range(tmp_indices.shape[0]):
                cur_tmp_indices = tmp_indices[ti]
                tensor_isin = torch.isin(cur_tmp_indices, indices_adv)  
                value_isin  = cur_tmp_indices[tensor_isin]  
                len_value   = value_isin.shape[0]
                print(len_value)
                mean_value.append(len_value + cur_tmp_indices.shape[0])
                assert cur_tmp_indices.shape[0] == topk
                
                adv_tmp_indices_y = torch.arange(len_value)
                adv_tmp_indices[ti][adv_tmp_indices_y] = value_isin + 1
            mean_value_t = torch.tensor(mean_value, dtype=torch.float).cuda()

        tot_num = tmp_indices.shape[0]
        epoch = tot_num//batch + (1 if tot_num % batch != 0 else 0)

        new_centers = None
        for i in range(epoch):
            start = i * batch
            end = i * batch + batch
            if end > tot_num:
                end = tot_num
            if strategy == 'none':
                if flip_z is None:
                    tmp_centers = torch.mean(torch.gather(matrix.unsqueeze(0).expand(end-start,-1,-1), 1, tmp_indices[start:end].unsqueeze(-1).expand(-1,-1,self.latent_size)),dim=1).squeeze()
                else:
                    ori_embs = torch.gather(matrix.unsqueeze(0).expand(end-start,-1,-1), 1, tmp_indices[start:end].unsqueeze(-1).expand(-1,-1,self.latent_size))     
                    new_embs = torch.gather(flip_z.unsqueeze(0).expand(end-start,-1,-1), 1, adv_tmp_indices[start:end].unsqueeze(-1).expand(-1,-1,self.latent_size)) 
                    all_embs = torch.cat((ori_embs, new_embs), dim=1)  
                    all_embs_sum = torch.sum(all_embs, dim=1) 
                    tmp_centers = all_embs_sum / mean_value_t[start:end].view(-1,1)

        return  new_centers
        


    def distance(self, matrix1, matrix2, batch=100):
       

        assert len(matrix1.shape) == 2
        assert len(matrix2.shape) == 2

        dis = None
        tot_num = matrix1.shape[0]
        epoch = tot_num//batch + (1 if tot_num % batch != 0 else 0)
        matrix1 = matrix1.unsqueeze(dim=1)
        matrix2 = matrix2.unsqueeze(dim=0)
        for i in range(epoch):
            start = i * batch
            end = i * batch + batch
            if end > tot_num:
                end = tot_num
            tmp_matrix1 = matrix1[start:end]
            tmp_dis = (tmp_matrix1 - matrix2) ** 2.0
            tmp_dis = torch.sum(tmp_dis, dim=-1).squeeze()
            if dis is None:
                dis = tmp_dis
            else:
                dis = torch.cat([dis, tmp_dis], dim=0)

        
        return dis

