import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

import traceback

from memory_block import MemoryBlock
from log import *


class MEM(nn.Module):
    def __init__(self, dim_M=3, debug=True):
        super(MEM, self).__init__()

        self.debug = debug

        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        featureExtractor = models.vgg16(pretrained=True)
        self.featureExtractor = nn.Sequential(*list(featureExtractor.children())[:-1])

        self.memoryBlock = MemoryBlock(dim_M=dim_M, debug=debug)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048 * dim_M, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=3),
        )

    def forward(self, x):
        if not torch.is_tensor(x):
            printDebug("Converting numpy array into torch tensor", debug=self.debug)
            x = torch.tensor(x).float()

        # x = x.to(self.device)
        printDebug(f"x: {x.shape}", debug=self.debug)
        x = x[0]
        feature = self.featureExtractor(x)
        feature = feature.view(feature.size(0), -1)
        printDebug(f"feature: {feature.shape}", debug=self.debug)

        x_hat = self.memoryBlock(feature)
        x_hat = x_hat.view(1, -1)
        printDebug(f"x_hat: {x_hat.shape}", debug=self.debug)
        out = self.classifier(x_hat)
        printDebug(f"out: {out.shape}", debug=self.debug)
        return out


if __name__ == "__main__":
    debug = True
    print(f"Debug mode: {debug}")

    global device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = MEM().to(device)

    sample = np.random.rand(10, 3, 244, 244)
    printDebug(f"Sample: {sample.shape}", debug=debug)

    try:
        pred = model(sample)
        printDebug(f"Output: {pred.shape}", debug=debug)
    except:
        printError(traceback.format_exc())
