import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from typing import List
from collections import OrderedDict
import random
import scanpy as sc

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.autograd as autograd
from torch.autograd import Variable

def sub_data_preprocess(adata: sc.AnnData, n_top_genes: int=1000, batch_key: str=None, flavor: str='seurat_v3', min_genes: int=200, min_cells: int=3) -> sc.AnnData:
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if flavor == 'seurat_v3':
        # count data is expected when flavor=='seurat_v3'
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor, batch_key=batch_key,span=0.5)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    if flavor != 'seurat_v3':
        # log-format data is expected when flavor!='seurat_v3'
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor, batch_key=batch_key,span=0.5)
    return adata



def data_preprocess(adata: sc.AnnData, key: str='batch', n_top_genes: int=2000, flavor: str='seurat_v3', min_genes: int=200, min_cells: int=3, n_batch: int=2) -> sc.AnnData:
    print('Establishing Adata for Next Step...')
    hv_adata = sub_data_preprocess(adata, n_top_genes=n_top_genes, batch_key = key, flavor=flavor, min_genes=min_genes, min_cells=min_cells)
    hv_adata = hv_adata[:, hv_adata.var['highly_variable']]
    print('PreProcess Done.')
    return hv_adata
 


def extract_data(data: sc.AnnData, key: str, batches, orders=None):
    adata_values = []
    adata_obs_list = []
    a = data.obs[key]

    for batch in batches:
        selected_data = data.X[data.obs[key] == batch]
        selected_data_array = selected_data.toarray()
        adata_values.append(selected_data_array)

        # 提取与当前批次相对应的观测元数据
        selected_obs = data.obs[data.obs[key] == batch]
        adata_obs_list.append(selected_obs)

    if orders is None:
        std_ = [np.sum(np.std(item, axis=0)) for item in adata_values]
        orders = np.argsort(std_)[::-1]
    else:
        orders = np.array([batches.index(item) for item in orders])

    return adata_values, orders, adata_obs_list



class ScDataset(Dataset):
    def __init__(self):
        self.dataset = []
        self.variable = None
        self.labels = None
        self.transform = None
        self.sample = None
        self.trees = []

    def __len__(self):
        return 10 * 1024

    def __getitem__(self, index):
        dataset_samples = []
        for j, dataset in enumerate(self.dataset):
            rindex1 = np.random.randint(len(dataset))
            rindex2 = np.random.randint(len(dataset))
            alpha = np.random.uniform(0, 1)
            sample = alpha*dataset[rindex1] + (1-alpha)*dataset[rindex2]
            dataset_samples.append(sample)
        return dataset_samples


########## PLOT FUNCTIONS ##########
def cat_data(data_A: np.float32, data_B: np.float32, labels: List[List[int]]=None):
    data = np.r_[data_A, data_B]
    if labels is None:
        label = np.zeros(len(data_A)+len(data_B))
        label[-len(data_B):] = 1
        label = np.array([label]).T
    else:
        label = np.r_[labels[0], labels[1]]
    return data, label


########## NEURAL NETWORK UTILITY ##########
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data_size, 1024),
            nn.BatchNorm1d(1024),
            Mish(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.relu = torch.nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Mish(),
            nn.Linear(512, 1024),
            Mish(),

        )
        self.decoder1 = nn.Sequential(
            nn.Linear(1024, data_size),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(1024, n_classes),
        )


    def forward(self, ec, es):
        x = self.decoder(torch.cat((ec, es), dim=-1))
        cell= self.decoder1(x)
        batch= self.decoder2(x)

        return self.relu(cell),self.relu(batch)





def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def BDACL_fast(
    adata,
    key = 'BATCH',
    n_epochs = 50,
    num_workers=0,
    lr = 0.0005,
    b1 = 0.5,
    b2 = 0.999,
    latent_dim = 256,
    n_critic = 5,
    lambda_co = 3,
    lambda_rc = 1,
    seed = 8,
    ):
    setup_seed(seed)
    a=adata.obs[key]
    batches = sorted(list(set(adata.obs[key])))
    scd = ScDataset()
    scd.variable = np.array(adata.var_names)
    adata_values, orders ,obs_values = extract_data(adata, key, batches, orders=None)
    obs_names = [np.array(adata.obs_names[adata.obs[key] == batch]) for batch in batches]  
    ec_obs_names = None
    for item in orders:
        if ec_obs_names is None:
            ec_obs_names = obs_names[item]
        else:
            ec_obs_names = np.r_[ec_obs_names, obs_names[item]]
    print('Step 1: Calibrating Celltype...')
    scd.dataset = [adata_values[i] for i in orders]
    dataloader = DataLoader(
        dataset = scd,
        batch_size=512,
        num_workers=num_workers,
    )
    global data_size
    global n_classes
    data_size = scd.dataset[0].shape[1]
    n_classes = len(scd.dataset)
    EC = Encoder(latent_dim)
    Dec = Decoder(latent_dim + n_classes)
    mse_loss = torch.nn.MSELoss()

    if cuda:
        EC.cuda()
        Dec.cuda()
        mse_loss.cuda()
    EC.apply(weights_init_normal)
    Dec.apply(weights_init_normal)
    optimizer_Dec = torch.optim.Adam(Dec.parameters(), lr=lr, betas=(b1, b2))
    optimizer_EC = torch.optim.Adam(EC.parameters(), lr=lr, betas=(b1, b2))
    for epoch in range(n_epochs):
        Dec.train()
        EC.train()
        for i, data in enumerate(dataloader):
            datum = [Variable(item.type(FloatTensor)) for item in data]
            batch_size = datum[0].shape[0]
            ES_data1 = -np.zeros((n_classes * batch_size, n_classes))
            for j in range(n_classes):
                ES_data1[j*batch_size:(j+1)*batch_size, j] = 1
            ES_data1 = Variable(torch.tensor(ES_data1).type(FloatTensor))
            optimizer_Dec.zero_grad()
            optimizer_EC.zero_grad()
            loss1_data1 = torch.cat(datum, dim=0)
            cell,batch=Dec(EC(loss1_data1), ES_data1)
            ae_loss = mse_loss(cell, loss1_data1)+0.4*mse_loss(batch, ES_data1)
            all_loss = ae_loss
            all_loss.backward()
            optimizer_Dec.step()
            optimizer_EC.step()
        # 打印损失信息
        print(
            "[Epoch %d/%d] [Reconstruction loss: %f]"
            % (epoch + 1, n_epochs, ae_loss.item())
        )
    Dec.eval()
    EC.eval()
    obs_ordered = [obs_values[i] for i in orders]


    with torch.no_grad():
        data = Variable(FloatTensor(scd.dataset[0]))
        label = np.full((len(scd.dataset[0]),1), batches[orders[0]])
        es_label = np.zeros((len(scd.dataset[0]), n_classes))
        es_label[:, orders[j]] = 1
        es_label_tensor = Variable(torch.tensor(es_label).type(FloatTensor))
        static_sample = EC(data)
        reconstructed_data_0,batch = Dec(static_sample, es_label_tensor)
        transform_data = reconstructed_data_0.cpu().detach().numpy()
        for j in range(1, len(scd.dataset)):
            data = Variable(FloatTensor(scd.dataset[j]))
            es_label = np.zeros((len(scd.dataset[j]), n_classes))
            es_label[:, orders[j]] = 1
            es_label_tensor = Variable(torch.tensor(es_label).type(FloatTensor))
            static_sample = EC(data)
            reconstructed_data_j,batch = Dec(static_sample, es_label_tensor)
            fake_data = reconstructed_data_j.cpu().detach().numpy()
            fake_label = np.full((len(scd.dataset[j]),1), batches[orders[j]])
            transform_data, label = cat_data(transform_data, fake_data, [label, fake_label])

    merged_obs = pd.concat(obs_ordered, axis=0)
    ec_data = sc.AnnData(transform_data)

    ec_data.obs = merged_obs
    ec_data.obs_names = ec_obs_names
    ec_data.obs[key] = label

    
    return EC, ec_data
