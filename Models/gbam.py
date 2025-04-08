import torch
import torch.nn as nn

class GlobalBandAmplification(nn.Module):
    def __init__(self, input_size=200):
        super(GlobalBandAmplification, self).__init__()
        self.input_size = input_size
        # 全局波段注意力网络
        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size // 16),  # 200 -> 12
            nn.ReLU(),
            nn.Linear(input_size // 16, input_size),  # 12 -> 200
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入：x [batch, 1, input_size]
        # 输出：w_global [batch, input_size]
        avg_pool = x.mean(dim=1)  # [batch, input_size]
        w_global = self.attention(avg_pool)
        return w_global