import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, x):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(self.kmeans_plus_plus_init(x))

    def kmeans_plus_plus_init(self, x):
        centers = torch.zeros(self.num_classes, x.size(1), device=x.device)

        first_center_idx = torch.randint(x.size(0), (1,))
        centers[0] = x[first_center_idx]

        for i in range(1, self.num_classes):
            distances = torch.min(torch.sum((x - centers[:i]) ** 2, dim=1), dim=0).values
            prob = distances / torch.sum(distances)

            # 将 prob 转换为张量，并保持其维度为 1 维
            prob_tensor = prob.unsqueeze(0)

            next_center_idx = torch.multinomial(prob_tensor, 1)
            centers[i] = x[next_center_idx]

        return centers

    def forward(self, x, labels):
        batch_size = x.size(0)

        # 计算特征向量 x 与其所属类别中心之间的距离矩阵
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers[labels], 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) - \
                  2 * torch.matmul(x, self.centers.t())

        # 对距离矩阵进行截断，防止出现除零错误
        distmat = distmat.clamp(min=1e-12, max=1e+12)

        # 根据标签计算每个样本与其所属类别中心之间的距离
        losses = distmat.gather(1, labels.view(-1, 1))

        # 计算损失，即距离的平均值
        loss = losses.mean()
        print("当前中心点的左边为={}".format(self.centers))
        return loss
