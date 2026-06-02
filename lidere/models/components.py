import torch
from torch import nn
import math


class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.lora_A = nn.Parameter(torch.zeros((r, linear.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((linear.out_features, r)))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # freeze base weights
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale