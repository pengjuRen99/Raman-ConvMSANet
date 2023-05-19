import torch
import torch.nn as nn
import torch.nn.functional as F



class LNGAPBlock(nn.Module):
    def __init__(self, in_features, num_classes, **kwargs):
        super().__init__()
        self.ln = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.ln(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x
    

class MLPBlock(nn.Module):
    def __init__(self, in_features, num_classes, **kwargs):
        super(MLPBlock, self).__init__()

        self.dense1 = nn.Linear(in_features, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.dense3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x
