'''
MLP_Mixer model

list
    MLP
    Mixer_Block
    MLP_Mixer

shinsjn
'''

import torch
import numpy as np
import torch.nn as nn

from einops.layers.torch import Reduce, Rearrange


class MLP(nn.Module):
    def __init__(self,dim,hidden_dim):
        super(MLP,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,dim)
        )

    def forward(self,x):
        return self.network(x)

class Mixer_Block(nn.Module):
    def __init__(self,patch_num,dim,token_dim,channel_dim):
        super(Mixer_Block, self).__init__()
        #MLP1 = token_mixer
        #MLP2 = channel_mixer

        self.MLP1 = nn.Sequential(
            nn.LayerNorm(dim),              # input tensor = (b,p,c) -> layernorm by last tensor
            Rearrange('b p c -> b c p'),    # transpose
            MLP(patch_num,token_dim),
            Rearrange('b c p -> b p c')
        )

        self.MLP2 = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim,channel_dim)
        )

    def forward(self,x):
        x = x + self.MLP1(x)                #Skip connection
        x = x + self.MLP2(x)
        return x

class MLP_Mixer(nn.Module):
    def __init__(self,dim,token_dim,channel_dim,patch_size,layer_depth,class_num,in_channels,input_size):
        super(MLP_Mixer, self).__init__()

        #patch embedding
        self.num_patch = (input_size // patch_size) ** 2

        self.per_patch_FC = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c')
        )


        #mixer_blocks

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(layer_depth):
            self.mixer_blocks.append(
                Mixer_Block(self.num_patch,dim,token_dim,channel_dim)
            )

        #layer norm -> github official code refer

        self.layer_norm = nn.LayerNorm(dim)

        self.final_mlp = nn.Linear(dim,class_num)

    def forward(self,x):

        x = self.per_patch_FC(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.final_mlp(x)

        return x


