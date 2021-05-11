import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

from log import *
from utils import clone_layer

debug = True


class MemoryUnit(nn.Module):
    def __init__(self, feature_size, embedding_size, debug=True):
        super(MemoryUnit, self).__init__()
        self.debug = debug

        self.embed = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=embedding_size), nn.Tanh()
        )

    def forward(self, x):
        x = self.embed(x)
        printDebug(f"MemUnit: {x.shape}", debug=self.debug)
        _x = x.T

        m = torch.mm(x, _x)

        prob = F.softmax(m, dim=1)
        prob = torch.mean(prob, axis=0)

        return prob


class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=kwargs["input_shape"], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=kwargs["output_shape"]),
            nn.ReLU(),
        )

    def forward(self, features):
        return self.net(features)


class MemoryBlock(nn.Module):
    def __init__(
        self,
        feature_size=25088,
        embedding_size=2048,
        bj_transform_size=2048,
        dim_M=3,
        debug=True,
    ):
        super(MemoryBlock, self).__init__()

        self.debug = debug

        self.autoEncoder = AutoEncoder(
            input_shape=feature_size, output_shape=bj_transform_size
        )
        self.dim_M = dim_M
        self.units = clone_layer(
            MemoryUnit(
                feature_size=feature_size, embedding_size=embedding_size, debug=debug
            ),
            dim_M,
        )

    def forward(self, x):
        printDebug(f"MemBlck: {x.shape}", debug=self.debug)
        C = self.autoEncoder(x)
        probs = []

        for unit in self.units:
            probs.append(unit(x).unsqueeze(0))

        X_hat = []
        for i in range(self.dim_M):
            X_hat.append(torch.matmul(probs[i], C))

        X_hat = torch.stack(X_hat, dim=0).squeeze(1)
        return X_hat
