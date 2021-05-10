import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

feature_size = 512 * 7 * 7
bj_transform_size = 2048
embedding_size = 2048


class MemoryUnit(nn.Module):
    def __init__(self):
        super(MemoryUnit, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=embedding_size), nn.Tanh()
        )

    def forward(self, x):
        x = self.embed(x)
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
    def __init__(self):
        super(MemoryBlock, self).__init__()
        self.autoEncoder = AutoEncoder(
            input_shape=feature_size, output_shape=bj_transform_size
        )
        self.memoryUnit1 = MemoryUnit()
        self.memoryUnit2 = MemoryUnit()
        self.memoryUnit3 = MemoryUnit()
        self.memoryUnit4 = MemoryUnit()
        self.memoryUnit5 = MemoryUnit()
        self.memoryUnit6 = MemoryUnit()
        self.memoryUnit7 = MemoryUnit()
        self.memoryUnit8 = MemoryUnit()
        self.memoryUnit9 = MemoryUnit()
        self.memoryUnit10 = MemoryUnit()

    def forward(self, x):
        C = self.autoEncoder(x)
        p1 = self.memoryUnit1(x).unsqueeze(0)
        p2 = self.memoryUnit2(x).unsqueeze(0)
        p3 = self.memoryUnit3(x).unsqueeze(0)
        p4 = self.memoryUnit4(x).unsqueeze(0)
        p5 = self.memoryUnit5(x).unsqueeze(0)
        p6 = self.memoryUnit6(x).unsqueeze(0)
        p7 = self.memoryUnit7(x).unsqueeze(0)
        p8 = self.memoryUnit8(x).unsqueeze(0)
        p9 = self.memoryUnit9(x).unsqueeze(0)
        p10 = self.memoryUnit10(x).unsqueeze(0)

        X_hat = []
        X_hat.append(torch.mm(p1, C))
        X_hat.append(torch.mm(p2, C))
        X_hat.append(torch.mm(p3, C))
        X_hat.append(torch.mm(p4, C))
        X_hat.append(torch.mm(p5, C))
        X_hat.append(torch.mm(p6, C))
        X_hat.append(torch.mm(p7, C))
        X_hat.append(torch.mm(p8, C))
        X_hat.append(torch.mm(p9, C))
        X_hat.append(torch.mm(p10, C))

        X_hat = torch.stack(X_hat, dim=0).squeeze(1)
        return X_hat
