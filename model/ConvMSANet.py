import numpy as np
import torch
import torch.nn as nn

from functools import partial
from timm.models.layers import DropPath
from einops import rearrange

import model.preresnet_dnn_block as preresnet_dnn
from model.attentions import Attention1d
import model.classifier_block as classifier


class LocalAttention(nn.Module):
    def __init__(self, dim_in, dim_out=None, *, window_size=5, k=1, heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        self.attn = Attention1d(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout,)
        self.window_size = window_size
        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h = x.shape
        p = self.window_size
        n = h // p

        mask = torch.zeros(p, p, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, 0], self.rel_index[:, 1]]

        x = rearrange(x, 'b c (n p) -> (b n) p c', p=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, '(b n) p c -> b c (n p)', n=n, p=p)
        return x, attn


    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x] for x in range(window_size)]))
        d = i[None, :] - i[:, None]     
        return d            # Relative position coding


class AttentionBlockB(nn.Module):
    expansion = 4
    
    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0, sd=0.0, stride=1, window_size=5, k=1, norm=nn.BatchNorm1d, activation=nn.GELU, **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion

        self.shortcut = []
        if stride != 1 or dim_in != dim_out * self.expansion:
            self.shortcut.appen(nn.Conv1d(dim_in, dim_out * self.expansion, stride=stride, bias=False))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.norm1 = norm(dim_in)
        self.relu = activation()
        self.conv = nn.Conv1d(dim_in, width, kernel_size=1, bias=False)
        self.norm2 = norm(width)
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout, )
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.norm1(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.conv(x)
        x = self.norm2(x)
        x, attn = self.attn(x)

        x = self.sd(x) + skip

        return x


class AttentionBasicBlockB(AttentionBlockB):
    expansion = 1


class Stem(nn.Module):
    def __init__(self, dim_in, dim_out, pool=True):
        super().__init__()
        self.layer0 = []
        if pool:
            self.layer0.append(nn.Conv1d(dim_in, dim_out, kernel_size=7, stride=2, padding=3, bias=False))  # 减小两倍
            self.layer0.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))    # 减小两倍
        else:
            self.layer0.append(nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False))
        self.layer0 = nn.Sequential(*self.layer0)

    def forward(self, x):
        x = self.layer0(x)
        return x            # 1000 -> 250


class ConvMSANet(nn.Module):
    def __init__(self, block1, block2, *, num_blocks, num_blocks2, heads, cblock=classifier.LNGAPBlock, window_size, sd=0.0, num_classes=10, stem=Stem, name, **block_kwargs):
        super().__init__()
        self.name = name
        idxs = [[j for j in range(sum(num_blocks[:i]), sum(num_blocks[:i + 1]))] for i in range(len(num_blocks))]
        sds = [[sd * j / (sum(num_blocks) - 1) for j in js] for js in idxs]

        self.layer0 = stem(1, 16)
        self.layer1 = self._make_layer(block1, block2, 16, 32, num_blocks[0], num_blocks2[0], stride=1, heads=heads[0], window_size=0, sds=sds[0], **block_kwargs)
        self.layer2 = self._make_layer(block1, block2, 32 * block2.expansion, 64, num_blocks[1], num_blocks2[1], stride=2, heads=heads[1], window_size=window_size[0], sds=sds[1], **block_kwargs)
        self.layer3 = self._make_layer(block1, block2, 64 * block2.expansion, 128, num_blocks[2], num_blocks2[2], stride=2, heads=heads[2], window_size=window_size[1], sds=sds[2], **block_kwargs)
        self.layer4 = self._make_layer(block1, block2, 128 * block2.expansion, 256, num_blocks[3], num_blocks2[3], stride=2, heads=heads[3], window_size=window_size[2], sds=sds[3], **block_kwargs)

        self.classifier = []
        if cblock is classifier.MLPBlock:
            self.classifier.append(nn.AdaptiveAvgPool1d((7)))
            self.classifier.append(cblock(7 * 256 * block2.expansion, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(256 * block2.expansion, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)

    @staticmethod
    def _make_layer(block1, block2, in_channels, out_channels, num_block1, num_block2, stride, heads, window_size, sds, **block_kwargs):
        alt_seq = [False] * (num_block1 - num_block2 * 2) + [False, True] * num_block2
        stride_seq = [stride] + [1] * (num_block1 - 1)

        seq, channels = [], in_channels
        for alt, stride, sd in zip(alt_seq, stride_seq, sds):
            block = block1 if not alt else block2       # False: Conv, True: multi-head self-attention
            seq.append(block(channels, out_channels, stride=stride, sd=sd, heads=heads, window_size=window_size, **block_kwargs))
            channels = out_channels * block.expansion

        return nn.Sequential(*seq)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x


def convmsa_reflection(num_classes=40, stem=True, name='ConvMSANet_Reflection', **block_kwargs):

    return ConvMSANet(preresnet_dnn.BasicBlock, AttentionBasicBlockB, stem=partial(Stem, pool=stem),
                    num_blocks=(2, 2, 2, 2), num_blocks2=(0, 1, 1, 1), heads=(3, 3, 6, 12), window_size=(25, 8, 4), sd=0.1,
                    num_classes=num_classes, name=name, **block_kwargs)


def convmsa_transmission(num_classes=42, stem=True, name='ConvMSANet_Transmission', **block_kwargs):

    return ConvMSANet(preresnet_dnn.BasicBlock, AttentionBasicBlockB, stem=partial(Stem, pool=stem),
                    num_blocks=(2, 2, 2, 2), num_blocks2=(0, 1, 1, 1), heads=(3, 2, 4, 8), window_size=(15, 10, 5), sd=0.1,
                    num_classes=num_classes, name=name, **block_kwargs)    
