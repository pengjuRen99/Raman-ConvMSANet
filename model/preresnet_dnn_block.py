import torch.nn as nn
from timm.models.layers import DropPath


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, groups=1, width_per_group=64, sd=0.0, **block_kwargs):
        super().__init__()

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        width = int(channels * (width_per_group / 64.)) * groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(nn.Conv1d(in_channels, channels, kernel_size=1, stride=stride, padding=0, bias=False))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()


    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.bn(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.bn(x)
            x = self.relu(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.sd(x) + skip

        return x