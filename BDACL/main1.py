from .Classifier import Classifier
from .data_preprocess import *
from .calculate_NN import get_dict_mnn,get_dict_mnn_para
from .utils import *
from .network import EmbeddingNet
from .logger import create_logger       ## import logger
from .pytorchtools import EarlyStopping ## import earlytopping

import os
from time import time
from scipy.sparse import issparse
from numpy.linalg import matrix_power
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners,reducers,distances



## 创建 scDML模型
"""
定义了一个 BDACL 模型对象，包括预处理、输入转换、相似性计算、维度缩减、评估数据集等方法。
"""
class scDMLModel:
    def __init__(self,verbose=True,save_dir="./results/"):
        """
        创建 scDMLModel 对象
        参数:
        verbose: 'str'，可选， 默认为 'True'，当 verbose=True 时，将额外的信息写入日志文件。
        save_dir: 保存结果和日志信息的文件夹
        """
        # 设置类属性 verbose，决定是否在日志中输出详细信息。
        self.verbose = verbose
        # 设置保存目录为类属性。
        self.save_dir = save_dir
        # 如果指定的保存目录不存在，则创建它。
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir + "/")
        # 创建一个日志文件来记录操作和信息。
        self.log = create_logger('', fh=self.save_dir + 'log.txt')
        # 如果 verbose 为 True，则在日志文件中写入更多的详细信息。
        if (self.verbose):
            # 记录已创建日志文件的消息。
            self.log.info("创建日志文件...")
            # 记录成功创建 scDMLModel 对象的消息。
            self.log.info("创建 Model 对象完成...")






            ###  预处理原始数据以生成初始聚类标签。
    "预处理原始数据集"
    def preprocess(self,adata,cluster_method="louvain",resolution=3.0,batch_key="BATCH",n_high_var = 1000,hvg_list=None,normalize_samples = True,target_sum=1e4,log_normalize = True,
                   normalize_features = True,pca_dim=100,scale_value=10.0,num_cluster=50,mode="unsupervised"):

        """
        预处理原始数据集
        参数:
        adata: anndata.AnnData，形状为 (n_obs, n_vars) 的带有注释的数据矩阵。行对应于细胞，列对应于基因。

        cluster_method: "str"，用于初始化细胞类型标签的聚类算法["louvain","leiden","kmeans","minibatch-kmeans"]

        resolution: 'np.float'，默认为3.0，louvain 算法的分辨率，用于初始化聚类。

        batch_key: str，观察数据框中标识每个细胞批次的列的名称。如果设置为 None，则假定所有细胞都来自一个批次。

        n_high_var: int，指定要识别为高变量基因的数量。例如，如果 n_high_var = 1000，则将具有最高方差的 1000 个基因指定为高变量基因。

        hvg_list: 'list'，高变量基因的列表，适用于 seqRNA 数据。

        normalize_samples: bool，如果为 True，在每个细胞中通过该细胞中表达计数的总和来对每个基因的表达进行归一化。

        target_sum: 'int'，默认为 1e4，细胞归一化后的总计数，您可以选择 1e6 进行 CPM 归一化。

        log_normalize: bool，如果为 True，对表达进行对数变换。即，对于每个基因、细胞表达计数，计算 log(expression + 1)。

        normalize_features: bool，如果为 True，对每个基因的表达进行 z-score 归一化。

        pca_dim: 'int'，主成分数量。

        scale_value: 用于 sc.pp.scale() 中的参数，用于截断异常值。

        num_cluster: "np.int"，kmeans 的 K 参数。

        返回:

        适用于 BDACL 集成的标准化 adata，在后续阶段中使用。
        """
        # 如果当前模式是"unsupervised"（无监督），则进行以下处理。
        if(mode == "unsupervised"):
            # 检查输入数据的完整性。
            batch_key = checkInput(adata, batch_key, self.log)
            # 设置 batch_key、分辨率和聚类方法作为类属性。
            self.batch_key = batch_key
            self.reso = resolution
            self.cluster_method = cluster_method
            # 获取批次的数量。
            self.nbatch = len(adata.obs[batch_key].value_counts())

            # 如果处于详细日志模式，输出一些预处理的参数信息。
            if(self.verbose):
                self.log.info("正在执行 preprocess() 函数...")
                self.log.info("模式={}".format(mode))
                self.log.info("聚类方法={}".format(cluster_method))
                self.log.info("分辨率={}".format(str(resolution)))
                self.log.info("批次键={}".format(str(batch_key)))

            # 设置正则化参数。
            self.norm_args = (batch_key, n_high_var, hvg_list, normalize_samples, target_sum, log_normalize, normalize_features, scale_value, self.verbose, self.log)
            # 调用 Normalization 函数进行数据标准化。
            normalized_adata = Normalization(adata, *self.norm_args)
            # 调用 dimension_reduction 函数进行降维。
            emb = dimension_reduction(normalized_adata, pca_dim, self.verbose, self.log)
            # 调用 init_clustering 函数进行初始聚类。
            init_clustering(emb, reso=self.reso, cluster_method=cluster_method, verbose=self.verbose, log=self.log)

            # 设置批次索引。
            self.batch_index = normalized_adata.obs[batch_key].values
            # 在标准化的数据中添加初始化聚类的结果。
            normalized_adata.obs["init_cluster"] = emb.obs["init_cluster"].values.copy()
            # 获取初始化聚类的数量。
            self.num_init_cluster = len(emb.obs["init_cluster"].value_counts())

            if(self.verbose):
                self.log.info("预处理数据集完成。")
            # 返回预处理后的数据。
            return normalized_adata

        # 如果当前模式是"supervised"（有监督），则进行以下处理。
        elif(mode == "supervised"):
            # 检查输入数据的完整性。
            batch_key = checkInput(adata, batch_key, self.log)
            # 设置 batch_key、分辨率和聚类方法作为类属性。
            self.batch_key = batch_key
            self.reso = resolution
            self.cluster_method = cluster_method

            # 设置正则化参数。
            self.norm_args = (batch_key, n_high_var, hvg_list, normalize_samples, target_sum, log_normalize, normalize_features, scale_value, self.verbose, self.log)
            # 调用 Normalization 函数进行数据标准化。
            normalized_adata = Normalization(adata, *self.norm_args)

            if(self.verbose):
                self.log.info("模式={}".format(mode))
                self.log.info("批次键={}".format(str(batch_key)))
                self.log.info("预处理数据集完成。")
            # 返回预处理后的数据。
            return normalized_adata






    # 定义一个方法，用于将归一化的 AnnData 数据转换为 BDACL 的训练数据。
    def convertInput(self, adata, batch_key="BATCH", celltype_key=None, mode="unsupervised"):
        """
        将归一化的 AnnData 数据转换为训练数据。

        参数:
        - adata: anndata.AnnData 对象，包含归一化后的数据。
        - batch_key: 字符串，指定在 adata.obs 中的批次标签，默认为 "BATCH"。
        - celltype_key: 字符串，指定细胞类型的键，默认为 None。
        - mode: 字符串，训练模式，可以是 "unsupervised"（无监督）或 "supervised"（有监督），默认为 "unsupervised"。
        """
        # 如果训练模式为 "unsupervised"（无监督）
        if(mode == "unsupervised"):
            # 检查输入数据的有效性。
            checkInput(adata, batch_key=batch_key, log=self.log)
            # 如果 AnnData 对象中没有 PCA 数据，则进行 PCA 分析。
            if("X_pca" not in adata.obsm.keys()):
                sc.tl.pca(adata)
            # 如果 AnnData 对象中没有初始聚类结果，则进行邻居搜索和聚类。
            if("init_cluster" not in adata.obs.columns):
                sc.pp.neighbors(adata, random_state=0)
                sc.tl.louvain(adata, key_added="init_cluster", resolution=3.0)
            # 如果数据是稀疏的，转换为数组格式。
            if(issparse(adata.X)):
                self.train_X = adata.X.toarray()
            else:
                self.train_X = adata.X.copy()
            # 记录批次数量。
            self.nbatch = len(adata.obs[batch_key].value_counts())
            # 保存初始聚类的结果。
            self.train_label = adata.obs["init_cluster"].values.copy()
            # 保存 PCA 嵌入矩阵。
            self.emb_matrix = adata.obsm["X_pca"].copy()
            # 保存批次索引。
            self.batch_index = adata.obs[batch_key].values
            # 创建一个数据框，保存初始聚类的结果。
            self.merge_df = pd.DataFrame(adata.obs["init_cluster"])
            # 如果处于详细日志模式，保存初始聚类的分布到文件。
            if(self.verbose):
                self.merge_df.value_counts().to_csv(self.save_dir + "cluster_distribution.csv")
            # 如果提供了细胞类型的键，保存细胞类型。
            if(celltype_key is not None):
                self.celltype = adata.obs[celltype_key].values
            else:
                self.celltype = None
        # 如果训练模式为 "supervised"（有监督）
        elif(mode == "supervised"):
            # 如果没有提供细胞类型的键，则抛出一个错误。
            if(celltype_key is None):
                self.log.info("请在监督模式下提供cell类型密钥 !")
                raise IOError
            # 如果数据是稀疏的，转换为数组格式。
            if(issparse(adata.X)):
                self.train_X = adata.X.toarray()
            else:
                self.train_X = adata.X.copy()
            # 保存细胞类型。
            self.celltype = adata.obs[celltype_key].values
            # 记录细胞类型的数量。
            self.ncluster = len(adata.obs[celltype_key].value_counts())
            # 创建一个空的数据框。
            self.merge_df = pd.DataFrame()
            # 将细胞类型转换为分类码，并保存到数据框中。
            self.merge_df["nc_" + str(self.ncluster)] = self.celltype
            self.merge_df["nc_" + str(self.ncluster)] = self.merge_df["nc_" + str(self.ncluster)].astype("category").cat.codes






    ### 计算连接性
    "计算聚类的连通性"

    # 定义一个方法，用于使用 KNN 和 MNN 计算 BDACL 簇之间的相似性。
    def calculate_similarity(self, K_in=5, K_bw=10, K_in_metric="cosine", K_bw_metric="cosine"):
        """
        使用 KNN 和 MNN 计算 BDACL 簇之间的相似性。

        参数:
        - K_in: 内部 KNN 的邻居数量。
        - K_bw: 计算 MNN 时的邻居数量。
        - K_in_metric: 内部 KNN 的距离度量标准。
        - K_bw_metric: MNN 的距离度量标准。
        """

        # 保存 K_in 和 K_bw 的值到对象的属性中。
        self.K_in = K_in
        self.K_bw = K_bw
        # 如果处于详细日志模式，打印 K_in 和 K_bw 的值。
        if (self.verbose):
            self.log.info("K_in={}, K_bw={}".format(K_in, K_bw))
            self.log.info("开始计算 KNN 和 MNN 以获取簇之间的相似性。")
        # 如果批次数量小于 10，则不使用并行计算。
        if (self.nbatch < 10):
            # 提示用户即将使用近似方法计算每个批次内的 KNN 对。
            if (self.verbose):
                self.log.info("使用近似方法计算每个批次内的 KNN 对...")
            # 调用 get_dict_mnn 函数计算并获取内部 KNN 对。
            knn_intra_batch_approx = get_dict_mnn(data_matrix=self.emb_matrix, batch_index=self.batch_index, k=K_in,
                                                  flag="in", metric=K_in_metric, approx=True, return_distance=False,
                                                  verbose=self.verbose, log=self.log)
            # 将近似的 KNN 对转换为 numpy 数组格式。
            knn_intra_batch = np.array([list(i) for i in knn_intra_batch_approx])
            # 提示用户即将使用近似方法计算批次之间的 MNN 对。
            if (self.verbose):
                self.log.info("使用近似方法计算批次之间的 MNN 对...")
            # 调用 get_dict_mnn 函数计算并获取批次之间的 MNN 对。
            mnn_inter_batch_approx = get_dict_mnn(data_matrix=self.emb_matrix, batch_index=self.batch_index, k=K_bw,
                                                  flag="out", metric=K_bw_metric, approx=True, return_distance=False,
                                                  verbose=self.verbose, log=self.log)
            # 将近似的 MNN 对转换为 numpy 数组格式。
            mnn_inter_batch = np.array([list(i) for i in mnn_inter_batch_approx])
            # 提示用户已完成所有最近邻居的查找。
            if (self.verbose):
                self.log.info("查找所有最近邻居完成。")
        # 如果批次数量大于或等于 10，使用并行模式进行计算，加速过程。
        else:
            # 提示用户即将在并行模式下进行计算。
            if (self.verbose):
                self.log.info("在并行模式下计算 KNN 和 MNN 对以加速计算。")
                self.log.info("使用近似方法计算每个批次内的 KNN 对...")
            # 使用并行方法获取内部 KNN 对。
            knn_intra_batch_approx = get_dict_mnn_para(data_matrix=self.emb_matrix, batch_index=self.batch_index,
                                                       k=K_in, flag="in", metric=K_in_metric, approx=True,
                                                       return_distance=False, verbose=self.verbose, log=self.log)
            # 将近似的 KNN 对转换为 numpy 数组格式。
            knn_intra_batch = np.array(knn_intra_batch_approx)
            # 提示用户即将使用近似方法计算批次之间的 MNN 对。
            if (self.verbose):
                self.log.info("使用近似方法计算批次之间的 MNN 对...")
            # 使用并行方法获取批次之间的 MNN 对。
            mnn_inter_batch_approx = get_dict_mnn_para(data_matrix=self.emb_matrix, batch_index=self.batch_index,
                                                       k=K_bw, flag="out", metric=K_bw_metric, approx=True,
                                                       return_distance=False, verbose=self.verbose, log=self.log)
            # 将近似的 MNN 对转换为 numpy 数组格式。
            mnn_inter_batch = np.array(mnn_inter_batch_approx)
            # 提示用户已完成所有最近邻居的查找。
            if (self.verbose):
                self.log.info("查找所有最近邻居完成。")
        # 提示用户即将计算簇之间的相似性矩阵。
        if (self.verbose):
            self.log.info("计算簇之间的相似性矩阵。")
            # 调用 cal_sim_matrix 函数，使用 KNN 和 MNN 结果计算簇之间的相似性矩阵和 NN 配对矩阵。
            # cor_matrix是相似性矩阵    nn_matrix是NN 配对矩阵
            self.cor_matrix, self.nn_matrix = cal_sim_matrix(knn_intra_batch, mnn_inter_batch, self.train_label,
                                                             self.verbose, self.log)
            # 如果处于详细日志模式，则保存相似性矩阵和 NN 配对矩阵到文件中。
            if (self.verbose):
                self.log.info("将相似性矩阵保存到文件中...")
                # 将相似性矩阵保存为 CSV 文件。
                self.cor_matrix.to_csv(self.save_dir + "cor_matrix.csv")
                self.log.info("将 nn 配对矩阵保存到文件中。")
                # 将 NN 配对矩阵保存为 CSV 文件。
                self.nn_matrix.to_csv(self.save_dir + "nn_matrix.csv")
                self.log.info("完成相似性矩阵计算。")
            # 如果提供了细胞类型信息，则进行以下分析。
            if (self.celltype is not None):
                # 检查 MNN 对中有多少是连接相同细胞类型的。
                same_celltype = self.celltype[mnn_inter_batch[:, 0]] == self.celltype[mnn_inter_batch[:, 1]]
                # 统计相同细胞类型的 MNN 对数量。
                equ_pair = sum(same_celltype)
                self.log.info("连接相同细胞类型的 MNN 配对的数量为 {}".format(equ_pair))
                # 计算连接相同细胞类型的 MNN 对的比率。
                equ_ratio = sum(self.celltype[mnn_inter_batch[:, 1]] == self.celltype[mnn_inter_batch[:, 0]]) / \
                            same_celltype.shape[0]
                self.log.info("连接相同细胞类型的 MNN 配对的比率为 {}".format(equ_ratio))
                # 创建一个数据框来保存 MNN 对中的细胞类型配对信息。
                df = pd.DataFrame({"celltype_pair1": self.celltype[mnn_inter_batch[:, 0]],
                                   "celltype_pair2": self.celltype[mnn_inter_batch[:, 1]]})
                # 获取每对细胞类型在 MNN 对中出现的次数。
                num_info = pd.crosstab(df["celltype_pair1"], df["celltype_pair2"], margins=True, margins_name="Total")
                # 获取每对细胞类型在 MNN 对中出现的行比率。
                ratio_info_row = pd.crosstab(df["celltype_pair1"], df["celltype_pair2"]).apply(lambda r: r / r.sum(),
                                                                                               axis=1)
                # 获取每对细胞类型在 MNN 对中出现的列比率。
                ratio_info_col = pd.crosstab(df["celltype_pair1"], df["celltype_pair2"]).apply(lambda r: r / r.sum(),
                                                                                               axis=0)
                # 保存细胞类型配对的数量信息和比率信息到 CSV 文件中。
                num_info.to_csv(self.save_dir + "mnn_pair_num_info.csv")
                ratio_info_row.to_csv(self.save_dir + "mnn_pair_ratio_info_raw.csv")
                ratio_info_col.to_csv(self.save_dir + "mnn_pair_ratio_info_col.csv")
                # 打印细胞类型配对的数量信息和比率信息。
                self.log.info(num_info)
                self.log.info(ratio_info_row)
                self.log.info(ratio_info_col)
            # 方法结束，返回内部 KNN 对、批次之间的 MNN 对、相似性矩阵和 NN 配对矩阵。
            return knn_intra_batch, mnn_inter_batch, self.cor_matrix, self.nn_matrix








    '''定义一个方法，用于合并小的聚类簇到较大的聚类簇，并重新分配簇的标签。'''
    def merge_cluster(self,ncluster_list=[3],merge_rule="rule2"):
        """
        将小簇合并到较大簇中并重新分配簇标签
        参数:
        ncluster_list: 'list'，您可以设置一个fixed_ncluster的列表，以观察scDML的合并过程。
        merger_rule: 'str'，默认为"rule2"，scDML实施了两种类型的合并规则，在大多数情况下，这两个规则会生成相同的结果。
        返回:
        merge_df: "pd.DataFrame"，具有不同簇数的合并簇的标签。
        """
        # 初始化一个DataFrame来保存聚类信息。
        self.nc_list=pd.DataFrame()
        # 将ncluster_list中的整数转换为字符串，以便于后续操作。
        dis_cluster=[str(i) for i in ncluster_list]
        # 创建一个合并数据框的深拷贝，以避免修改原始数据。
        df=self.merge_df.copy()
        # 添加一个全为1的新列"value"。
        df["value"]=np.ones(self.train_X.shape[0])
        # 如果处于详细日志模式，则记录正在使用的合并规则。
        if(self.verbose):
            self.log.info("BDACL merge cluster with "+merge_rule+"....")
        # 如果选择的是"rule1"合并规则
        if(merge_rule=="rule1"):
            # 遍历希望的聚类数量列表
            for n_cluster in ncluster_list:
                # 使用第一种规则进行聚类合并，并获取合并后的映射集合。
                map_set=merge_rule1(self.cor_matrix.copy(),self.num_init_cluster,n_cluster=n_cluster,save_dir=self.save_dir)
                # 初始化一个空字典来保存聚类的映射关系。
                map_dict={}
                # 遍历映射集合
                for index,item in enumerate(map_set):
                    # 对于每个集合中的聚类
                    for c in item:
                        # 将当前聚类标签映射到新的聚类编号。
                        map_dict[str(c)]=index
                # 使用映射字典更新合并数据框中的聚类标签
                self.merge_df["nc_"+str(n_cluster)]=self.merge_df["init_cluster"].map(map_dict)
                # 在数据框中添加一个新列来保存新的聚类标签。
                df[str(n_cluster)]=str(n_cluster)+"("+self.merge_df["nc_"+str(n_cluster)].astype(str)+")"
                # 如果处于详细日志模式，则记录当前的聚类合并集合。
                if(self.verbose):
                    self.log.info("merging cluster set:"+str(map_set)) #
        # 如果选择的是"rule2"合并规则
        if(merge_rule=="rule2"):
            # 遍历希望的聚类数量列表
            for n_cluster in ncluster_list:
                # 使用第二种规则进行聚类合并，并获取合并后的映射集合。
                map_set=merge_rule2(self.cor_matrix.copy(),self.nn_matrix.copy(),self.merge_df["init_cluster"].value_counts().values.copy(),n_cluster=n_cluster,verbose=self.verbose,log=self.log)
                # 初始化一个空字典来保存聚类的映射关系。
                map_dict={}
                # 遍历映射集合
                for index,item in enumerate(map_set):
                    # 对于每个集合中的聚类
                    for c in item:
                        # 将当前聚类标签映射到新的聚类编号。
                        map_dict[str(c)]=index
                # 使用映射字典更新合并数据框中的聚类标签。
                self.merge_df["nc_"+str(n_cluster)]=self.merge_df["init_cluster"].map(map_dict)
                # 在数据框中添加一个新列来保存新的聚类标签。
                df[str(n_cluster)]=str(n_cluster)+"("+self.merge_df["nc_"+str(n_cluster)].astype(str)+")"
                # 如果处于详细日志模式，则记录当前的聚类合并集合。
                if(self.verbose):
                    self.log.info("merging cluster set:"+str(map_set)) #
        return df






    "为 BDACL 训练构建网络"
    def build_net(self, in_dim=1000,out_dim=32,emb_dim=[256],projection=False,project_dim=2,use_dropout=False,
            dp_list=None,use_bn=False, actn=nn.ReLU(),seed=1029):
        """
        构建用于scDML训练的网络
        参数:
        in_dim: 默认为1000，嵌入网络的输入维度，应与adata的高变量基因数量相等
        out_dim: 默认为32，嵌入网络的输出维度（即嵌入维度）
        emb_dim: 默认为[256]，隐藏层的维度
        projection: 默认为False，如果为True，则构建嵌入维度为project_dim（2或3）的投影网络
        project_dim: 默认为2，投影网络的输出维度
        use_drop: 默认为False，如果use_drop为True，则嵌入网络将添加DropOut层
        dp_list: 默认为None，当use_drop为True且嵌入网络具有两个以上的隐藏层时，可以设置与DropOut层一致的一组丢失率值的列表
        use_bn: 默认为False，如果use_bn为True，则嵌入网络将应用批归一化
        actn: 默认为nn.ReLU()，嵌入网络的激活函数
        seed: 默认为1029，这是用于PyTorch的随机种子，用于重现结果
        """
        # 检查输入维度是否与训练数据的特征数匹配
        if(in_dim != self.train_X.shape[1]):
            in_dim = self.train_X.shape[1]
        # 如果设置为详细日志模式，则记录信息
        if(self.verbose):
            self.log.info("为scDML培训构建嵌入网络")
        # 设置随机种子，以确保结果的可复现性
        seed_torch(seed)
        # 构建嵌入网络模型
        self.model=EmbeddingNet(in_sz=in_dim,out_sz=out_dim,emb_szs=emb_dim,projection=projection,project_dim=project_dim,dp_list
                                =dp_list,use_bn=use_bn,actn=actn)
        # 如果设置为详细日志模式，则显示模型的结构，并记录网络构建完成的信息
        if(self.verbose):
            self.log.info(self.model)
            self.log.info("构建嵌入网络完成…")





    "使用三元组损失训练 BDACL"
    def train(self,expect_num_cluster=None,merge_rule="rule2",num_epochs=100,batch_size=64,early_stop=False,patience=5,delta=50,
              metric="euclidean",margin=0.2,triplet_type="hard",device=None,save_model=False,mode="unsupervised"):
        """
        使用三元组损失训练scDML
        参数:
        -expect_num_cluster: 默认为None，您希望scDML合并的预期聚类数。如果此参数为None，
        scDML将使用默认阈值确定要合并的聚类数。

        -merge_rule: 默认为"relu2"，在expect_num_cluster为None时，scDML用于合并聚类的合并规则。

        -num_epochs: 默认为50。训练scDML的最大迭代次数。

        -early_stop: 当early_stop为True时，嵌入网络将根据早停规则停止训练。

        -patience: 默认为5。在上次验证损失改善后等待的时间长度。

        -delta: 默认为50，监测数量（困难三元组的数量）的最小变化，才被视为改善。

        -metric: 默认为字符串"euclidean"，用于计算三元组损失的距离类型。

        -margin: 用于计算三元组损失的超参数。

        -triplet_type: 将用于挖掘和优化三元组损失的三元组类型。

        -save_model: 是否在scDML训练后保存模型。

        -mode: 如果mode=="unsupervised"，scDML将使用相似性矩阵的合并规则结果。
        如果mode=="supervised"，scDML将使用细胞类型的真实标签来整合数据集。
        返回:
        -embedding: 默认为AnnData，经过scDML训练后进行批效应消除的AnnData数据。
        """



        '''
        算法3部分，得到聚类数量
        '''
        '如果训练模式为“unsupervised”（非监督），则根据数据相似性矩阵的合并规则结果训练模型'
        if(mode=="unsupervised"):
            # 检查是否已经提供了预期的聚类数量。如果没有，该方法会尝试自动确定聚类数量。
            if(expect_num_cluster is None):
                # 检查是否开启了详细模式(verbose)。如果开启，该方法会输出更多的日志信息。
                if(self.verbose):
                    # 输出日志信息，告知用户由于没有提供预期的聚类数量，因此将使用特征值差距来估计细胞类型的数量。
                    self.log.info("expect_num_cluster为None，使用特征值差距来估计细胞类型......的数量 ")
                # 从当前对象中获取相似性矩阵，并创建其副本。
                cor_matrix=self.cor_matrix.copy()
                # 遍历相似性矩阵的每一行。
                for i in range(len(cor_matrix)):
                    # 将相似性矩阵的对角线上的值设置为0，因为一个样本与自己的相似度应该是0。
                    cor_matrix.loc[i,i]=0.0

                    # 将相似性矩阵标准化到[0,1]范围内，方法是将矩阵的每个值除以矩阵的最大值。
                    A=cor_matrix.values/np.max(cor_matrix.values)# normalize similarity matrix to [0,1]
    
                    # 增强相似性结构
                    norm_A=A+matrix_power(A,2)
                    # # 对增强后的相似性矩阵的每一行进行迭代。
                    for i in range(len(A)):
                        # 再次确保矩阵的对角线上的值为0。
                        norm_A[i,i]=0.0
                    #cor_matrix
                # # 使用特征值分解方法来估计最优的聚类数量。
                k, _,  _ = eigenDecomposition(norm_A,save_dir=self.save_dir)
                # # 记录日志信息，显示估计的最佳聚类数量
                self.log.info(f'最优簇数是 {k}')
                #  默认选择第一个最佳聚类数量作为预期的聚类数量。
                expect_num_cluster=k[0]



            # 检查是否已经有对应预期聚类数量的合并结果。
            if("nc_"+str(expect_num_cluster) not in self.merge_df):
                # 如果没有找到对应的合并结果，记录日志信息提醒用户。
                self.log.info("scDML找不到集群的mering结果={} ,您可以运行合并集群(fixed_ncluster={}) 函数来得到这个".format(expect_num_cluster,expect_num_cluster))
                # 由于没有找到预期的合并结果，所以抛出一个IOError。
                raise IOError
            # 从合并结果中获取训练标签。
            self.train_label=self.merge_df["nc_"+str(expect_num_cluster)].values.astype(int)



        # 如果训练模式为“supervised”（监督），则使用细胞类型的真实标签来整合数据集
        elif(mode=="supervised"):
            # 设置预期的聚类数量为真实的聚类数量。
            expect_num_cluster=self.ncluster
            self.train_label=self.merge_df["nc_"+str(expect_num_cluster)].values.astype(int)




        # 如果既不是'unsupervised'模式，也不是'supervised'模式，进入此分支。
        else:
            # 记录日志信息，表示当前模式未实现。
            self.log.info("未实现!!!")
            raise IOError








        # 检查指定目录下是否已存在训练好的模型文件
        if os.path.isfile(os.path.join(self.save_dir,"scDML_model.pkl")):
            # 如果存在，记录日志表示正在加载已训练的模型
            self.log.info("加载训练模型...")
            # 从指定路径加载模型文件到self.model中
            self.model=torch.load(os.path.join(self.save_dir,"scDML_model.pkl"))
        # 如果不存在已训练的模型文件，则进入训练流程
        else:
            # 如果处于详细日志模式
            if(self.verbose):
                # 记录日志，表示使用Embedding Net训练scDML
                self.log.info("train BDACL(expect_num_cluster={}) with Embedding Net".format(expect_num_cluster))
                # 记录期望的聚类数量
                self.log.info("expect_num_cluster={}".format(expect_num_cluster))
            # 如果没有指定训练设备（如GPU或CPU）
            if(device is None):
                # 自动检测是否有可用的GPU，有则使用GPU，否则使用CPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # 如果处于详细日志模式
                if(self.verbose):
                    # 如果GPU可用
                    if(torch.cuda.is_available()):
                        # 记录日志，表示使用GPU训练模型
                        self.log.info("利用GPU对模型进行训练")
                    # 如果GPU不可用
                    else:
                        # 记录日志，表示使用CPU训练模型
                        self.log.info("利用CPU训练模型")


            # 创建一个Tensor数据集，其中包括训练数据和其对应的标签
            train_set = torch.utils.data.TensorDataset(torch.FloatTensor(self.train_X), torch.from_numpy(self.train_label).long())
            # 使用DataLoader加载数据集，指定批大小、工作线程数和是否随机洗牌
            train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0,shuffle=True)
            # 将模型移动到指定的训练设备上（GPU或CPU）
            self.model=self.model.to(device)
            # 定义优化器为Adam，并设置学习率为0.01
            optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            # 如果指定的距离度量为余弦相似度
            if(metric=="cosine"):
                # 使用余弦相似度作为距离度量
                distance = distances.CosineSimilarity()# use cosine_similarity()
            # 如果指定的距离度量为欧氏距离
            elif(metric=="euclidean"):
                # 使用欧氏距离作为距离度量
                distance=distances.LpDistance(p=2,normalize_embeddings=False) # use euclidean distance
            # 如果指定的距离度量既不是余弦相似度也不是欧氏距离
            else:
                # 记录日志，表示当前指定的距离度量未实现
                self.log.info("未实现，有待更新")
                raise IOError
            #reducer: reduce the loss between all triplet(mean)
            # 定义一个阈值规约器，当损失低于0时，将其设置为0
            reducer = reducers.ThresholdReducer(low = 0)
            #Define Loss function
            # 定义损失函数为三元组间隔损失，其中使用之前定义的距离度量和规约器，并设置间隔为给定的margin
            loss_func = losses.TripletMarginLoss(margin = margin, distance = distance, reducer = reducer)
            #Define miner_function
            # 定义挖掘函数为三元组间隔挖掘器，使用之前定义的距离度量，并设置间隔为给定的margin和三元组的类型
            mining_func = miners.TripletMarginMiner(margin = margin, distance = distance, type_of_triplets = triplet_type)


            # 初始化损失函数
            num_classes = expect_num_cluster  # 有3个细胞类型
            feat_dim = 32  # 嵌入维度为32
            alpha = 0.5  # 中心损失的权重
            # 假设有10个类别，嵌入维度为32
            cross_entropy_loss = nn.CrossEntropyLoss()
            classifier = Classifier(feat_dim, num_classes).to(device)  # 注意将模型移动到相应的设备上


            # 如果处于详细日志模式
            if(self.verbose):
                # 记录日志，表示使用指定的距离度量和三元组类型进行模型训练
                self.log.info("use {} distance and {} triplet to train model".format(metric,triplet_type))
            # 初始化一个空数组，用于存储每个时期挖掘的困难三元组的数量
            mined_epoch_triplet=np.array([])
            # 如果不使用早停
            if(not early_stop):
                # 如果处于详细日志模式
                if(self.verbose):
                    # 记录日志，表示不使用早停
                    self.log.info("not use earlystopping!!!!")
                # 对指定的迭代次数进行迭代
                for epoch in range(1, num_epochs+1):
                    # 初始化当前迭代的损失为0
                    temp_epoch_loss=0
                    # 初始化当前迭代挖掘的困难三元组的数量为0
                    temp_num_triplet=0
                    # 将模型设置为训练模式
                    self.model.train()
                    # 对训练数据进行批次迭代
                    for batch_idx, (train_data, training_labels) in enumerate(train_loader):
                        # 将训练数据和标签移动到指定的训练设备上（GPU或CPU）
                        train_data, training_labels = train_data.to(device), training_labels.to(device)
                        # 在每次迭代开始前，重置优化器的梯度为0
                        optimizer.zero_grad()
                        "输入数据 train_data 被传递给 self.model 进行前向运算，生成 embeddings"
                        # 使用模型对当前批次的训练数据进行前向运算，生成嵌入向量
                        embeddings = self.model(train_data)
                        # 使用挖掘函数对嵌入向量进行挖掘，得到困难三元组的索引
                        indices_tuple = mining_func(embeddings, training_labels)
                        # 使用损失函数计算当前批次的损失
                        loss = loss_func(embeddings, training_labels, indices_tuple)
                        output = classifier(embeddings)  # 假设有一个分类器层
                        loss1 = cross_entropy_loss(output, training_labels)
                        # 累加当前批次的困难三元组数量
                        temp_num_triplet=temp_num_triplet+indices_tuple[0].size(0)
                        # 对损失进行反向传播，计算梯度
                        loss = loss1 + alpha * loss
                        temp_epoch_loss = temp_epoch_loss+loss
                        # 对损失进行反向传播，计算梯度
                        loss.backward()
                        # 使用优化器更新模型参数
                        optimizer.step()

                    # 将当前迭代挖掘的困难三元组数量添加到数组中，用于跟踪每个迭代的进度
                    mined_epoch_triplet=np.append(mined_epoch_triplet,temp_num_triplet)
                    # 如果处于详细日志模式
                    if(self.verbose):
                        # 记录日志，表示当前迭代的困难三元组数量
                        self.log.info("epoch={},number_hard_triplet={},loss={}".format(epoch,temp_num_triplet,temp_epoch_loss))
            # 如果使用早停
            else:
                # 如果处于详细日志模式
                if(self.verbose):
                    # 记录日志，表示使用早停
                    self.log.info("use earlystopping!!!!")
                # 初始化早停对象，设置容忍度、最小改进量、保存路径和日志函数
                early_stopping = EarlyStopping(patience=patience, delta=delta,verbose=True,path=self.save_dir+"checkpoint.pt",trace_func=self.log.info)
                # 对指定的迭代次数进行迭代
                for epoch in range(1, num_epochs+1):
                    # 初始化当前迭代的损失为0
                    temp_epoch_loss=0
                    # 初始化当前迭代挖掘的困难三元组的数量为0
                    temp_num_triplet=0
                    # 将模型设置为训练模式
                    self.model.train()
                    # 对训练数据进行批次迭代
                    for batch_idx, (train_data, training_labels) in enumerate(train_loader):
                        # 将训练数据和标签移动到指定的训练设备上（GPU或CPU）
                        train_data, training_labels = train_data.to(device), training_labels.to(device)
                        # 在每次迭代开始前，重置优化器的梯度为0
                        optimizer.zero_grad()
                        # 使用模型对当前批次的训练数据进行前向运算，生成嵌入向量
                        embeddings = self.model(train_data)
                        # 使用挖掘函数对嵌入向量进行挖掘，得到困难三元组的索引
                        indices_tuple = mining_func(embeddings, training_labels)
                        # 使用损失函数计算当前批次的损失
                        loss = loss_func(embeddings, training_labels, indices_tuple)
                        # 累加当前批次的困难三元组数量
                        temp_num_triplet=temp_num_triplet+indices_tuple[0].size(0)

                        # 对损失进行反向传播，计算梯度
                        loss.backward()
                        # 使用优化器更新模型参数
                        optimizer.step()
                    # 使用早停对象检查是否应该提前停止训练，基于当前迭代的困难三元组数量
                    early_stopping(temp_num_triplet, self.model)

                    #检查是否满足早停的条件。如果满足，训练将被终止
                    if early_stopping.early_stop:
                        # 输出日志信息，告知用户训练由于满足了早停条件而被终止。
                        self.log.info("Early stopping")
                        break

                    mined_epoch_triplet=np.append(mined_epoch_triplet,temp_num_triplet)
                    # 再次检查是否开启了详细模式。
                    if(self.verbose):
                        self.log.info("epoch={},number_hard_triplet={}".format(epoch,temp_num_triplet))
            if(self.verbose):
                # 输出日志信息，告知用户scDML的训练已经完成。
                self.log.info("BDACL training done....")
            ##### save embedding model
            # 检查是否需要保存训练后的模型。
            if(save_model):
                if(self.verbose):
                    # 输出日志信息，告知用户模型正在被保存。
                    self.log.info("save model....")
                # 将模型保存到指定的目录。
                torch.save(self.model.to(torch.device("cpu")),os.path.join(self.save_dir,"scDML_model.pkl"))
            # 更新当前对象的损失属性，使其等于训练过程中挖掘的三元组数量。
            self.loss=mined_epoch_triplet
        ##### generate embeding
        # 使用训练好的模型对训练数据进行预测，生成特征嵌入。
        features=self.predict(self.train_X)
        return features





    def predict(self, X, batch_size=128):
        """
        对数据矩阵进行预测以生成嵌入。

        参数:
        X: 待预测的数据矩阵。
        batch_size: 用于数据加载的批次大小。
        """
        # 如果设置为详细模式，则在日志中记录开始预测的信息。
        if (self.verbose):
            self.log.info("extract embedding for dataset with trained network")
        # 设置设备为 CPU。
        device = torch.device("cpu")
        # 创建一个数据加载器，用于按批次加载数据。
        dataloader = DataLoader(
            torch.FloatTensor(X), batch_size=batch_size, pin_memory=False, shuffle=False
        )
        # 创建一个迭代器，用于按批次迭代数据，并显示一个进度条。
        data_iterator = tqdm(dataloader, leave=False, unit="batch")
        # 将模型移到指定的设备上。
        self.model = self.model.to(device)
        # 初始化一个空列表，用于存储模型的输出特征。
        features = []
        # 确保不会计算梯度，因为我们只是进行预测。
        with torch.no_grad():
            # 将模型设置为评估模式。
            self.model.eval()
            # 对数据迭代器中的每个批次进行迭代。
            for batch in data_iterator:
                # 将批次数据移到指定的设备上。
                batch = batch.to(device)
                # 输入批次数据到模型中进行前向运算，生成输出。
                output = self.model(batch)
                # 将输出从 GPU 移到 CPU，并添加到特征列表中。
                features.append(
                    output.detach().cpu()
                )
            # 将所有的特征合并为一个 numpy 数组。
            features = torch.cat(features).cpu().numpy()
        # 返回嵌入特征。
        return features





    ##### scDML的集成
    "使用 BDACL 进行批次对齐以进行集成"
    def integrate(self,adata,batch_key="BATCH",ncluster_list=[3],expect_num_cluster=None,K_in=6,K_bw=12,K_in_metric="cosine",K_bw_metric="cosine",merge_rule="rule2",num_epochs=100,
                  projection=False,early_stop=False,batch_size=64,metric="euclidean",margin=0.2,triplet_type="hard",device=None,seed=1029,out_dim=32,emb_dim=[256],save_model=False,celltype_key=None,mode="unsupervised"):
        """
        使用scDML进行批次对齐集成
        参数:
        adata: 经过归一化的AnnData数据
        celltype_key: 用于评估mnn配对的比率或在监督模式下使用的细胞类型键
        mode: 默认为字符串，"unsupervised"，用户可以选择"unsupervised"或"supervised"
        ...
        ...
        ...
        """
        # 在日志中记录当前的模式（监督或非监督）。
        self.log.info("mode={}".format(mode))
        #start_time=time()
        # 如果模式为非监督。
        if(mode=="unsupervised"):
            # 将给定的数据 adata 转换为训练数据。
            self.convertInput(adata,batch_key=batch_key,celltype_key=celltype_key,mode=mode)
            #print("convert input...cost time={}s".format(time()-start_time))
            # 计算簇之间的相似性。
            self.calculate_similarity(K_in=K_in,K_bw=K_bw,K_in_metric=K_in_metric,K_bw_metric=K_bw_metric)
            #print("calculate similarity matrix done...cost time={}s".format(time()-start_time))
            # 合并簇并重新分配簇标签。
            self.merge_cluster(ncluster_list=ncluster_list,merge_rule=merge_rule)
            #print("reassign cluster label done...cost time={}s".format(time()-start_time))
            # 为 BDACL 构建嵌入网络。
            self.build_net(out_dim=out_dim,emb_dim=emb_dim,projection=projection,seed=seed)
            #print("construct network done...cost time={}s".format(time()-start_time))
            # 训练 BDACL 以消除批次效应。
            features=self.train(expect_num_cluster=expect_num_cluster,num_epochs=num_epochs,early_stop=early_stop,batch_size=batch_size,metric=metric,margin=margin,triplet_type=triplet_type,device=device,save_model=save_model,mode=mode)
            #print("train neural network done...cost time={}s".format(time()-start_time)) 
            # save result
        # 如果模式为监督。
        elif(mode=="supervised"):
            # 将给定的数据 adata 转换为训练数据。
            self.convertInput(adata,batch_key=batch_key,celltype_key=celltype_key,mode=mode)
            # 为 BDACL 构建嵌入网络。
            self.build_net(out_dim=out_dim,emb_dim=emb_dim,projection=projection,seed=seed)
            #print("construct network done...cost time={}s".format(time()-start_time))
            # 训练 BDACL 以消除批次效应。
            features=self.train(expect_num_cluster=expect_num_cluster,num_epochs=num_epochs,early_stop=early_stop,batch_size=batch_size,metric=metric,margin=margin,triplet_type=triplet_type,device=device,save_model=save_model,mode=mode)
            #print("train neural network done...cost time={}s".format(time()-start_time)) 
            # save result
        # 如果模式既不是监督也不是非监督，则引发错误。
        else:
            self.log.info("Not implemented!!!")
            raise IOError
        # 将计算出的特征赋值给 adata 的 obsm 属性。
        adata.obsm["X_emb"]=features
        # 将重新分配的簇标签赋值给 adata 的 obs 属性。
        adata.obs["reassign_cluster"]=self.train_label.astype(int).astype(str)
        adata.obs["reassign_cluster"]=adata.obs["reassign_cluster"].astype("category")


        

        
    

