from .Classifier import Classifier
from .data_preprocess import *
from .calculate_NN import get_dict_mnn, get_dict_mnn_para
from .utils import *
from .network import EmbeddingNet
from .logger import create_logger  ## import logger
from .pytorchtools import EarlyStopping  ## import earlytopping

import os
from time import time
from scipy.sparse import issparse
from numpy.linalg import matrix_power
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners, reducers, distances


import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import BDACL
import sys
print(sys.version)
from BDACL.utils import print_dataset_information
import scanpy as sc
import ED
import copy

class BDACLModel:
    def __init__(self, verbose=True, save_dir="./results/"):

        self.verbose = verbose
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir + "/")
        self.log = create_logger('', fh=self.save_dir + 'log.txt')
        if (self.verbose):
            self.log.info("创建日志文件...")
            self.log.info("创建 Model 对象完成...")


    def preprocess(self, adata, cluster_method="louvain", resolution=3.0, batch_key="BATCH", n_high_var=1000,
                   hvg_list=None, normalize_samples=True, target_sum=1e4, log_normalize=True,
                   normalize_features=True, pca_dim=100, scale_value=10.0, num_cluster=50, mode="unsupervised"):

        if (mode == "unsupervised"):
            batch_key = checkInput(adata, batch_key, self.log)
            self.batch_key = batch_key
            self.reso = resolution
            self.cluster_method = cluster_method
            self.nbatch = len(adata.obs[batch_key].value_counts())

            if (self.verbose):
                self.log.info("正在执行 preprocess() 函数...")
                self.log.info("模式={}".format(mode))
                self.log.info("聚类方法={}".format(cluster_method))
                self.log.info("分辨率={}".format(str(resolution)))
                self.log.info("批次键={}".format(str(batch_key)))

            self.norm_args = (
            batch_key, n_high_var, hvg_list, normalize_samples, target_sum, log_normalize, normalize_features,
            scale_value, self.verbose, self.log)
            normalized_adata = Normalization(adata, *self.norm_args)
            emb = dimension_reduction(normalized_adata, pca_dim, self.verbose, self.log)
            init_clustering(emb, reso=self.reso, cluster_method=cluster_method, verbose=self.verbose, log=self.log)

            self.batch_index = normalized_adata.obs[batch_key].values
            normalized_adata.obs["init_cluster"] = emb.obs["init_cluster"].values.copy()
            self.num_init_cluster = len(emb.obs["init_cluster"].value_counts())

            if (self.verbose):
                self.log.info("预处理数据集完成。")
            return normalized_adata

        elif (mode == "supervised"):
            batch_key = checkInput(adata, batch_key, self.log)
            self.batch_key = batch_key
            self.reso = resolution
            self.cluster_method = cluster_method

            self.norm_args = (
            batch_key, n_high_var, hvg_list, normalize_samples, target_sum, log_normalize, normalize_features,
            scale_value, self.verbose, self.log)
            normalized_adata = Normalization(adata, *self.norm_args)

            if (self.verbose):
                self.log.info("模式={}".format(mode))
                self.log.info("批次键={}".format(str(batch_key)))
                self.log.info("预处理数据集完成。")
            return normalized_adata


    def convertInput(self, adata, batch_key="BATCH", celltype_key=None, mode="unsupervised"):

        if (mode == "unsupervised"):
            checkInput(adata, batch_key=batch_key, log=self.log)
            if ("X_pca" not in adata.obsm.keys()):
                sc.tl.pca(adata)
            if ("init_cluster" not in adata.obs.columns):
                sc.pp.neighbors(adata, random_state=0)
                sc.tl.louvain(adata, key_added="init_cluster", resolution=3.0)
            if (issparse(adata.X)):
                self.train_X = adata.X.toarray()
            else:
                self.train_X = adata.X.copy()
            self.nbatch = len(adata.obs[batch_key].value_counts())
            self.train_label = adata.obs["init_cluster"].values.copy()
            self.emb_matrix = adata.obsm["X_pca"].copy()
            self.batch_index = adata.obs[batch_key].values
            self.merge_df = pd.DataFrame(adata.obs["init_cluster"])
            if (self.verbose):
                self.merge_df.value_counts().to_csv(self.save_dir + "cluster_distribution.csv")
            if (celltype_key is not None):
                self.celltype = adata.obs[celltype_key].values
            else:
                self.celltype = None
        elif (mode == "supervised"):
            if (celltype_key is None):
                self.log.info("请在监督模式下提供cell类型密钥 !")
                raise IOError
            if (issparse(adata.X)):
                self.train_X = adata.X.toarray()
            else:
                self.train_X = adata.X.copy()
            self.celltype = adata.obs[celltype_key].values
            self.ncluster = len(adata.obs[celltype_key].value_counts())
            self.merge_df = pd.DataFrame()
            self.merge_df["nc_" + str(self.ncluster)] = self.celltype
            self.merge_df["nc_" + str(self.ncluster)] = self.merge_df["nc_" + str(self.ncluster)].astype(
                "category").cat.codes


    def calculate_similarity(self, K_in=5, K_bw=10, K_in_metric="cosine", K_bw_metric="cosine"):

        self.K_in = K_in
        self.K_bw = K_bw
        if (self.verbose):
            self.log.info("K_in={}, K_bw={}".format(K_in, K_bw))
            self.log.info("开始计算 KNN 和 MNN 以获取簇之间的相似性。")
        if (self.nbatch < 10):
            if (self.verbose):
                self.log.info("使用近似方法计算每个批次内的 KNN 对...")
            knn_intra_batch_approx = get_dict_mnn(data_matrix=self.emb_matrix, batch_index=self.batch_index, k=K_in,
                                                  flag="in", metric=K_in_metric, approx=True, return_distance=False,
                                                  verbose=self.verbose, log=self.log)
            knn_intra_batch = np.array([list(i) for i in knn_intra_batch_approx])
            if (self.verbose):
                self.log.info("使用近似方法计算批次之间的 MNN 对...")
            mnn_inter_batch_approx = get_dict_mnn(data_matrix=self.emb_matrix, batch_index=self.batch_index, k=K_bw,
                                                  flag="out", metric=K_bw_metric, approx=True, return_distance=False,
                                                  verbose=self.verbose, log=self.log)
            mnn_inter_batch = np.array([list(i) for i in mnn_inter_batch_approx])
            if (self.verbose):
                self.log.info("查找所有最近邻居完成。")
        else:
            if (self.verbose):
                self.log.info("在并行模式下计算 KNN 和 MNN 对以加速计算。")
                self.log.info("使用近似方法计算每个批次内的 KNN 对...")
            knn_intra_batch_approx = get_dict_mnn_para(data_matrix=self.emb_matrix, batch_index=self.batch_index,
                                                       k=K_in, flag="in", metric=K_in_metric, approx=True,
                                                       return_distance=False, verbose=self.verbose, log=self.log)
            knn_intra_batch = np.array(knn_intra_batch_approx)
            if (self.verbose):
                self.log.info("使用近似方法计算批次之间的 MNN 对...")
            mnn_inter_batch_approx = get_dict_mnn_para(data_matrix=self.emb_matrix, batch_index=self.batch_index,
                                                       k=K_bw, flag="out", metric=K_bw_metric, approx=True,
                                                       return_distance=False, verbose=self.verbose, log=self.log)
            mnn_inter_batch = np.array(mnn_inter_batch_approx)
            if (self.verbose):
                self.log.info("查找所有最近邻居完成。")
        if (self.verbose):
            self.log.info("计算簇之间的相似性矩阵。")
            self.cor_matrix, self.nn_matrix = cal_sim_matrix(knn_intra_batch, mnn_inter_batch, self.train_label,
                                                             self.verbose, self.log)
            if (self.verbose):
                self.log.info("将相似性矩阵保存到文件中...")
                self.cor_matrix.to_csv(self.save_dir + "cor_matrix.csv")
                self.log.info("将 nn 配对矩阵保存到文件中。")
                self.nn_matrix.to_csv(self.save_dir + "nn_matrix.csv")
                self.log.info("完成相似性矩阵计算。")
            if (self.celltype is not None):
                same_celltype = self.celltype[mnn_inter_batch[:, 0]] == self.celltype[mnn_inter_batch[:, 1]]
                equ_pair = sum(same_celltype)
                self.log.info("连接相同细胞类型的 MNN 配对的数量为 {}".format(equ_pair))
                equ_ratio = sum(self.celltype[mnn_inter_batch[:, 1]] == self.celltype[mnn_inter_batch[:, 0]]) / \
                            same_celltype.shape[0]
                self.log.info("连接相同细胞类型的 MNN 配对的比率为 {}".format(equ_ratio))
                df = pd.DataFrame({"celltype_pair1": self.celltype[mnn_inter_batch[:, 0]],
                                   "celltype_pair2": self.celltype[mnn_inter_batch[:, 1]]})
                num_info = pd.crosstab(df["celltype_pair1"], df["celltype_pair2"], margins=True, margins_name="Total")
                ratio_info_row = pd.crosstab(df["celltype_pair1"], df["celltype_pair2"]).apply(lambda r: r / r.sum(),
                                                                                               axis=1)
                ratio_info_col = pd.crosstab(df["celltype_pair1"], df["celltype_pair2"]).apply(lambda r: r / r.sum(),
                                                                                               axis=0)
                num_info.to_csv(self.save_dir + "mnn_pair_num_info.csv")
                ratio_info_row.to_csv(self.save_dir + "mnn_pair_ratio_info_raw.csv")
                ratio_info_col.to_csv(self.save_dir + "mnn_pair_ratio_info_col.csv")
                self.log.info(num_info)
                self.log.info(ratio_info_row)
                self.log.info(ratio_info_col)
            return knn_intra_batch, mnn_inter_batch, self.cor_matrix, self.nn_matrix


    def merge_cluster(self, ncluster_list=[3], merge_rule="rule2"):
        self.nc_list = pd.DataFrame()
        dis_cluster = [str(i) for i in ncluster_list]
        df = self.merge_df.copy()
        df["value"] = np.ones(self.train_X.shape[0])
        if (self.verbose):
            self.log.info("BDACL merge cluster with " + merge_rule + "....")
        if (merge_rule == "rule1"):
            for n_cluster in ncluster_list:
                map_set = merge_rule1(self.cor_matrix.copy(), self.num_init_cluster, n_cluster=n_cluster,
                                      save_dir=self.save_dir)
                map_dict = {}
                for index, item in enumerate(map_set):
                    for c in item:
                        map_dict[str(c)] = index
                self.merge_df["nc_" + str(n_cluster)] = self.merge_df["init_cluster"].map(map_dict)
                df[str(n_cluster)] = str(n_cluster) + "(" + self.merge_df["nc_" + str(n_cluster)].astype(str) + ")"
                if (self.verbose):
                    self.log.info("merging cluster set:" + str(map_set))  #
        if (merge_rule == "rule2"):
            for n_cluster in ncluster_list:
                map_set = merge_rule2(self.cor_matrix.copy(), self.nn_matrix.copy(),
                                      self.merge_df["init_cluster"].value_counts().values.copy(), n_cluster=n_cluster,
                                      verbose=self.verbose, log=self.log)
                map_dict = {}
                for index, item in enumerate(map_set):
                    for c in item:
                        map_dict[str(c)] = index
                self.merge_df["nc_" + str(n_cluster)] = self.merge_df["init_cluster"].map(map_dict)
                df[str(n_cluster)] = str(n_cluster) + "(" + self.merge_df["nc_" + str(n_cluster)].astype(str) + ")"
                if (self.verbose):
                    self.log.info("merging cluster set:" + str(map_set))  #
        return df



    def build_net(self, in_dim=1000, out_dim=32, emb_dim=[256], projection=False, project_dim=2, use_dropout=False,
                  dp_list=None, use_bn=False, actn=nn.ReLU(), seed=1029):
        # 检查输入维度是否与训练数据的特征数匹配
        if (in_dim != self.train_X.shape[1]):
            in_dim = self.train_X.shape[1]
        if (self.verbose):
            self.log.info("为BDACL培训构建嵌入网络")
        seed_torch(seed)
        self.model = EmbeddingNet(in_sz=in_dim, out_sz=out_dim, emb_szs=emb_dim, projection=projection,
                                  project_dim=project_dim, dp_list
                                  =dp_list, use_bn=use_bn, actn=actn)
        if (self.verbose):
            self.log.info(self.model)
            self.log.info("构建嵌入网络完成…")

    def train(self, expect_num_cluster=None, merge_rule="rule2", num_epochs=50, batch_size=64, early_stop=False,
              patience=5, delta=50,
              metric="euclidean", margin=0.2, triplet_type="hard", device=None, save_model=False, mode="unsupervised", alpha=0.5):
        if (mode == "unsupervised"):
            if (expect_num_cluster is None):
                if (self.verbose):
                    self.log.info("expect_num_cluster为None，使用特征值差距来估计细胞类型......的数量 ")
                cor_matrix = self.cor_matrix.copy()
                for i in range(len(cor_matrix)):
                    cor_matrix.loc[i, i] = 0.0
                    A = cor_matrix.values / np.max(cor_matrix.values)  # normalize similarity matrix to [0,1]
                    norm_A = A + matrix_power(A, 2)
                    for i in range(len(A)):
                        norm_A[i, i] = 0.0
                k, _, _ = eigenDecomposition(norm_A, save_dir=self.save_dir)
                self.log.info(f'最优簇数是 {k}')
                expect_num_cluster = k[0]
            if ("nc_" + str(expect_num_cluster) not in self.merge_df):
                self.log.info(
                    "BDACL找不到集群的mering结果={} ,您可以运行合并集群(fixed_ncluster={}) 函数来得到这个".format(
                        expect_num_cluster, expect_num_cluster))
                raise IOError
            self.train_label = self.merge_df["nc_" + str(expect_num_cluster)].values.astype(int)
        elif (mode == "supervised"):
            # 设置预期的聚类数量为真实的聚类数量。
            expect_num_cluster = self.ncluster
            self.train_label = self.merge_df["nc_" + str(expect_num_cluster)].values.astype(int)
        else:
            self.log.info("未实现!!!")
            raise IOError

        if os.path.isfile(os.path.join(self.save_dir, "BDACL_model.pkl")):
            self.log.info("加载训练模型...")
            self.model = torch.load(os.path.join(self.save_dir, "BDACL_model.pkl"))
        else:
            if (self.verbose):
                self.log.info("train BDACL(expect_num_cluster={}) with Embedding Net".format(expect_num_cluster))
                self.log.info("expect_num_cluster={}".format(expect_num_cluster))
            if (device is None):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if (self.verbose):
                    if (torch.cuda.is_available()):
                        self.log.info("利用GPU对模型进行训练")
                    else:
                        self.log.info("利用CPU训练模型")

            train_set = torch.utils.data.TensorDataset(torch.FloatTensor(self.train_X),
                                                       torch.from_numpy(self.train_label).long())
            train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
            self.model = self.model.to(device)
            optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            if (metric == "cosine"):
                distance = distances.CosineSimilarity()  # use cosine_similarity()
            elif (metric == "euclidean"):
                distance = distances.LpDistance(p=2, normalize_embeddings=False)  # use euclidean distance
            else:
                self.log.info("未实现，有待更新")
                raise IOError
            reducer = reducers.ThresholdReducer(low=0)
            loss_func = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducer)
            mining_func = miners.TripletMarginMiner(margin=margin, distance=distance, type_of_triplets=triplet_type)
            num_classes = expect_num_cluster  # 有3个细胞类型
            feat_dim = 32  # 嵌入维度为32
            cross_entropy_loss = nn.CrossEntropyLoss()
            classifier = Classifier(feat_dim, num_classes).to(device)
            mined_epoch_triplet = np.array([])
            if (not early_stop):
                if (self.verbose):
                    self.log.info("not use earlystopping!!!!")
                for epoch in range(1, num_epochs + 1):
                    temp_epoch_loss = 0
                    temp_num_triplet = 0
                    self.model.train()
                    for batch_idx, (train_data, training_labels) in enumerate(train_loader):
                        train_data, training_labels = train_data.to(device), training_labels.to(device)
                        optimizer.zero_grad()
                        embeddings = self.model(train_data)
                        indices_tuple = mining_func(embeddings, training_labels)
                        loss = loss_func(embeddings, training_labels, indices_tuple)
                        output = classifier(embeddings)
                        loss1 = cross_entropy_loss(output, training_labels)
                        temp_num_triplet=temp_num_triplet+indices_tuple[0].size(0)
                        loss = loss + alpha * loss1
                        temp_epoch_loss = temp_epoch_loss+loss
                        loss.backward()
                        optimizer.step()

                    mined_epoch_triplet = np.append(mined_epoch_triplet, temp_num_triplet)
                    if (self.verbose):
                        self.log.info("epoch={},loss={}".format(epoch, temp_num_triplet/100))
            # 如果使用早停
            else:
                if (self.verbose):
                    self.log.info("use earlystopping!!!!")
                early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True,
                                               path=self.save_dir + "checkpoint.pt", trace_func=self.log.info)
                for epoch in range(1, num_epochs + 1):
                    # 初始化当前迭代的损失为0
                    temp_epoch_loss = 0
                    temp_num_triplet = 0
                    self.model.train()
                    for batch_idx, (train_data, training_labels) in enumerate(train_loader):
                        train_data, training_labels = train_data.to(device), training_labels.to(device)
                        optimizer.zero_grad()
                        embeddings = self.model(train_data)
                        indices_tuple = mining_func(embeddings, training_labels)
                        loss = loss_func(embeddings, training_labels, indices_tuple)
                        temp_num_triplet = temp_num_triplet + indices_tuple[0].size(0)

                        loss.backward()
                        optimizer.step()
                    early_stopping(temp_num_triplet, self.model)

                    if early_stopping.early_stop:
                        self.log.info("Early stopping")
                        break

                    mined_epoch_triplet = np.append(mined_epoch_triplet, temp_num_triplet)
                    if (self.verbose):
                        self.log.info("epoch={}".format(epoch))
            if (self.verbose):
                self.log.info("BDACL training done....")
            if (save_model):
                if (self.verbose):
                    self.log.info("save model....")
                torch.save(self.model.to(torch.device("cpu")), os.path.join(self.save_dir, "BDACL_model.pkl"))
            self.loss = mined_epoch_triplet/100
        features = self.predict(self.train_X)
        return features

    def predict(self, X, batch_size=128):
        if (self.verbose):
            self.log.info("extract embedding for dataset with trained network")
        device = torch.device("cpu")
        dataloader = DataLoader(
            torch.FloatTensor(X), batch_size=batch_size, pin_memory=False, shuffle=False
        )
        data_iterator = tqdm(dataloader, leave=False, unit="batch")
        self.model = self.model.to(device)
        features = []
        with torch.no_grad():
            self.model.eval()
            for batch in data_iterator:
                batch = batch.to(device)
                output = self.model(batch)
                features.append(
                    output.detach().cpu()
                )
            features = torch.cat(features).cpu().numpy()
        return features



    def integrate(self, adata, batch_key="BATCH", ncluster_list=[3], expect_num_cluster=None, K_in=6, K_bw=12,
                  K_in_metric="cosine", K_bw_metric="cosine", merge_rule="rule2", num_epochs=100,
                  projection=False, early_stop=False, batch_size=64, metric="euclidean", margin=0.2,
                  triplet_type="hard", device=None, seed=1029, out_dim=32, emb_dim=[256], save_model=False,
                  celltype_key=None, mode="unsupervised"):
        self.log.info("mode={}".format(mode))
        if (mode == "unsupervised"):
            self.convertInput(adata, batch_key=batch_key, celltype_key=celltype_key, mode=mode)
            self.calculate_similarity(K_in=K_in, K_bw=K_bw, K_in_metric=K_in_metric, K_bw_metric=K_bw_metric)
            self.merge_cluster(ncluster_list=ncluster_list, merge_rule=merge_rule)
            self.build_net(out_dim=out_dim, emb_dim=emb_dim, projection=projection, seed=seed)
            features = self.train(expect_num_cluster=expect_num_cluster, num_epochs=num_epochs, early_stop=early_stop,
                                  batch_size=batch_size, metric=metric, margin=margin, triplet_type=triplet_type,
                                  device=device, save_model=save_model, mode=mode)
        elif (mode == "supervised"):
            self.convertInput(adata, batch_key=batch_key, celltype_key=celltype_key, mode=mode)
            self.build_net(out_dim=out_dim, emb_dim=emb_dim, projection=projection, seed=seed)
            features = self.train(expect_num_cluster=expect_num_cluster, num_epochs=num_epochs, early_stop=early_stop,
                                  batch_size=batch_size, metric=metric, margin=margin, triplet_type=triplet_type,
                                  device=device, save_model=save_model, mode=mode)
        else:
            self.log.info("Not implemented!!!")
            raise IOError
        adata.obsm["X_emb"] = features
        adata.obs["reassign_cluster"] = self.train_label.astype(int).astype(str)
        adata.obs["reassign_cluster"] = adata.obs["reassign_cluster"].astype("category")

    def deintegrate(self, adata, ncluster=None, batch_key="BATCH", expect_num_cluster=None, K_in=6, K_bw=12,
                  K_in_metric="cosine", K_bw_metric="cosine", merge_rule="rule2", num_epochs=100,
                  projection=False, early_stop=False, batch_size=64, metric="euclidean", margin=0.2,
                  triplet_type="hard", device=None, seed=1029, out_dim=32, emb_dim=[256], save_model=False,
                  celltype_key=None, mode="unsupervised"):
        print_dataset_information(adata,batch_key="BATCH",celltype_key="celltype")
        adata_copy = copy.deepcopy(adata)
        sc.pp.normalize_total(adata_copy,target_sum=1e4)
        sc.pp.log1p(adata_copy)
        sc.pp.highly_variable_genes(adata_copy,n_top_genes=1000,subset=True)
        sc.pp.scale(adata_copy)
        sc.tl.pca(adata_copy)
        sc.pp.neighbors(adata_copy)
        sc.tl.umap(adata_copy)
        sc.pl.umap(adata_copy,color=["BATCH","celltype"])
        adata_copy = copy.deepcopy(adata)
        adata__raw = ED.stage1.data_preprocess(adata_copy, 'BATCH')
        ED.stage1.BDACL_fast(adata__raw, key="BATCH")
        dataset = "newbact"
        save_dir = "./test_result/macaque_raw+/"
        BDACL = BDACLModel(save_dir=save_dir)
        adata = BDACL.preprocess(adata, cluster_method="louvain", resolution=3.0)
        if (mode == "unsupervised"):
            self.convertInput(adata, batch_key=batch_key, celltype_key=celltype_key, mode=mode)
            self.calculate_similarity(K_in=K_in, K_bw=K_bw, K_in_metric=K_in_metric, K_bw_metric=K_bw_metric)
            if (ncluster is None):
                if (self.verbose):
                    self.log.info("expect_num_cluster为None，使用特征值差距来估计细胞类型......的数量 ")
                cor_matrix = self.cor_matrix.copy()
                for i in range(len(cor_matrix)):
                    cor_matrix.loc[i, i] = 0.0

                    A = cor_matrix.values / np.max(cor_matrix.values)  # normalize similarity matrix to [0,1]

                    norm_A = A + matrix_power(A, 2)
                    for i in range(len(A)):
                        norm_A[i, i] = 0.0
                # # 使用特征值分解方法来估计最优的聚类数量。
                k, _, _ = eigenDecomposition(norm_A, save_dir=self.save_dir)
                self.log.info(f'最优簇数是 {k}')
                expect_num_cluster = k[0]
            self.merge_cluster(ncluster_list=[expect_num_cluster], merge_rule=merge_rule)
            self.build_net(out_dim=out_dim, emb_dim=emb_dim, projection=projection, seed=seed)
            features = self.train(expect_num_cluster=expect_num_cluster, num_epochs=num_epochs, early_stop=early_stop,
                                  batch_size=batch_size, metric=metric, margin=margin, triplet_type=triplet_type,
                                  device=device, save_model=save_model, mode=mode)
            adata.obsm["X_emb"] = features
            adata.obs["reassign_cluster"] = self.train_label.astype(int).astype(str)
            adata.obs["reassign_cluster"] = adata.obs["reassign_cluster"].astype("category")
        else:
            BDACL = BDACLModel(save_dir=save_dir)
            BDACL.integrate(adata, batch_key="BATCH", ncluster_list=[ncluster],
                            expect_num_cluster=ncluster, merge_rule="rule2")

        return adata







