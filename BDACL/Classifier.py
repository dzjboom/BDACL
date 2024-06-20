import torch.nn as nn


# 分类器
class Classifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)



