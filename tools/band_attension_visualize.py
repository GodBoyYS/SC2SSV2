import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# 模拟高光谱数据（假设输入为 145x145x200 的 IP 数据集）
# 这里我们简化数据为一个小的批次：batch_size=16, channels=200, height=1, width=1
batch_size = 16
num_channels = 200
num_classes = 16

# 模拟类别标签（16 个类别）
labels = torch.randint(0, num_classes, (batch_size,))

# 模拟输入数据
input_data = torch.randn(batch_size, num_channels, 1, 1)

# 定义普通网络（无波段注意力）
class SimpleSpectralCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleSpectralCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x.view(x.size(0), -1)  # 展平为 (batch_size, out_channels)

# 定义类别特异性波段注意力机制
class BandAttention(nn.Module):
    def __init__(self, in_channels):
        super(BandAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, in_channels, 1, 1)
        x_flat = x.view(x.size(0), x.size(1))  # (batch_size, in_channels)
        attention_weights = self.attention(x_flat)  # (batch_size, in_channels)
        attention_weights = attention_weights.view(x.size(0), x.size(1), 1, 1)  # 恢复维度
        x_weighted = x * attention_weights  # 应用注意力权重
        return x_weighted

# 定义全局波段注意力机制
class GlobalBandAttention(nn.Module):
    def __init__(self, in_channels):
        super(GlobalBandAttention, self).__init__()
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.global_attention(x)  # (batch_size, in_channels, 1, 1)
        x_weighted = x * attention_weights  # 应用全局注意力权重
        return x_weighted

# 组合网络
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.band_attention = BandAttention(in_channels)
        self.global_attention = GlobalBandAttention(in_channels)

    def forward(self, x, mode="simple"):
        if mode == "simple":
            # 普通网络
            x = self.conv(x)
        elif mode == "band_attention":
            # 类别特异性波段注意力
            x = self.band_attention(x)
            x = self.conv(x)
        elif mode == "global_attention":
            # 全局波段注意力
            x = self.global_attention(x)
            x = self.conv(x)
        x = self.relu(x)
        return x.view(x.size(0), -1)  # 展平为 (batch_size, out_channels)

# 初始化模型
model = FeatureExtractor(in_channels=num_channels, out_channels=64)

# 获取三种特征
model.eval()
with torch.no_grad():
    # 普通网络特征
    simple_features = model(input_data, mode="simple").numpy()
    # 类别特异性波段注意力特征
    band_attention_features = model(input_data, mode="band_attention").numpy()
    # 全局波段注意力特征
    global_attention_features = model(input_data, mode="global_attention").numpy()

# 使用 t-SNE 降维到 2D 进行可视化
# 设置 perplexity 小于 n_samples（batch_size=16）
perplexity_value = min(5, batch_size - 1)  # 确保 perplexity 小于样本数量
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)

# 降维
try:
    simple_features_2d = tsne.fit_transform(simple_features)
    band_attention_features_2d = tsne.fit_transform(band_attention_features)
    global_attention_features_2d = tsne.fit_transform(global_attention_features)
except ValueError as e:
    print(f"Error in t-SNE: {e}")
    print(f"Adjusting perplexity to a smaller value...")
    perplexity_value = min(perplexity_value - 1, batch_size - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
    simple_features_2d = tsne.fit_transform(simple_features)
    band_attention_features_2d = tsne.fit_transform(band_attention_features)
    global_attention_features_2d = tsne.fit_transform(global_attention_features)

# 可视化
plt.figure(figsize=(18, 5))

# 第一张图：普通网络特征
plt.subplot(1, 3, 1)
for i in range(num_classes):
    mask = labels.numpy() == i
    plt.scatter(simple_features_2d[mask, 0], simple_features_2d[mask, 1], label=f'Class {i}', alpha=0.6)
plt.title("Simple Network Features")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()

# 第二张图：类别特异性波段注意力特征
plt.subplot(1, 3, 2)
for i in range(num_classes):
    mask = labels.numpy() == i
    plt.scatter(band_attention_features_2d[mask, 0], band_attention_features_2d[mask, 1], label=f'Class {i}', alpha=0.6)
plt.title("Band Attention Features")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()

# 第三张图：全局波段注意力特征
plt.subplot(1, 3, 3)
for i in range(num_classes):
    mask = labels.numpy() == i
    plt.scatter(global_attention_features_2d[mask, 0], global_attention_features_2d[mask, 1], label=f'Class {i}', alpha=0.6)
plt.title("Global Band Attention Features")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()

plt.tight_layout()
plt.savefig("band_attention_comparison.png")
plt.show()