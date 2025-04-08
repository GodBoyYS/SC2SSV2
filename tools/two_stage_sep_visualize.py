import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# 模拟数据
num_classes = 16
k_shot = 5  # 每个类别的支持样本数
num_support = num_classes * k_shot  # 总支持样本数
num_channels = 64  # 特征维度（假设经过特征提取后的维度）

# 模拟支持样本和标签
support_data = torch.randn(num_support, num_channels)  # (80, 64)
support_labels = torch.repeat_interleave(torch.arange(num_classes), k_shot)  # 每个类别 5 个样本

# 定义原型网络
class PrototypicalNetwork(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(PrototypicalNetwork, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    def compute_prototypes(self, features, labels):
        prototypes = torch.zeros(self.num_classes, self.feature_dim)
        for c in range(self.num_classes):
            mask = labels == c
            prototypes[c] = features[mask].mean(dim=0)
        return prototypes

    def forward(self, support_features, support_labels, stage="simple"):
        # 计算初始原型
        prototypes = self.compute_prototypes(support_features, support_labels)

        if stage == "simple":
            return prototypes

        # 第一阶段：大样本混淆类别分离
        if stage == "stage1":
            # 模拟混淆矩阵（假设 Class 0 和 Class 1 混淆）
            confused_pairs = [(0, 1)]
            m1 = 3.0  # 分离边界
            for c1, c2 in confused_pairs:
                direction = prototypes[c1] - prototypes[c2]
                norm = torch.norm(direction)
                if norm > 0:
                    direction = direction / norm
                    prototypes[c1] += m1 * direction
                    prototypes[c2] -= m1 * direction
            return prototypes

        # 第二阶段：小样本类别分离
        if stage == "stage2":
            # 模拟小样本类别（假设 Class 2 和 Class 3 是小样本类别）
            small_classes = [2, 3]
            m2 = 5.0  # 分离边界
            for c in small_classes:
                other_prototypes = prototypes[torch.arange(self.num_classes) != c]
                mean_other = other_prototypes.mean(dim=0)
                direction = prototypes[c] - mean_other
                norm = torch.norm(direction)
                if norm > 0:
                    direction = direction / norm
                    prototypes[c] += m2 * direction
            return prototypes

# 初始化模型
model = PrototypicalNetwork(num_classes=num_classes, feature_dim=num_channels)

# 获取三种原型分布
model.eval()
with torch.no_grad():
    # 普通原型网络
    simple_prototypes = model(support_data, support_labels, stage="simple").numpy()
    # 第一阶段原型分离
    stage1_prototypes = model(support_data, support_labels, stage="stage1").numpy()
    # 第二阶段原型分离
    stage2_prototypes = model(support_data, support_labels, stage="stage2").numpy()

# 使用 t-SNE 降维到 2D 进行可视化
# 设置 perplexity 小于 n_samples（num_classes=16）
perplexity_value = min(5, num_classes - 1)  # 确保 perplexity 小于样本数量
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)

# 降维
try:
    simple_prototypes_2d = tsne.fit_transform(simple_prototypes)
    stage1_prototypes_2d = tsne.fit_transform(stage1_prototypes)
    stage2_prototypes_2d = tsne.fit_transform(stage2_prototypes)
except ValueError as e:
    print(f"Error in t-SNE: {e}")
    print(f"Adjusting perplexity to a smaller value...")
    perplexity_value = min(perplexity_value - 1, num_classes - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
    simple_prototypes_2d = tsne.fit_transform(simple_prototypes)
    stage1_prototypes_2d = tsne.fit_transform(stage1_prototypes)
    stage2_prototypes_2d = tsne.fit_transform(stage2_prototypes)

# 可视化
plt.figure(figsize=(18, 5))

# 第一张图：普通原型网络
plt.subplot(1, 3, 1)
plt.scatter(simple_prototypes_2d[:, 0], simple_prototypes_2d[:, 1], c=np.arange(num_classes), cmap='tab20', s=100)
for i in range(num_classes):
    plt.text(simple_prototypes_2d[i, 0], simple_prototypes_2d[i, 1], f'Class {i}', fontsize=8)
plt.title("Simple Prototypical Network")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# 第二张图：第一阶段原型分离
plt.subplot(1, 3, 2)
plt.scatter(stage1_prototypes_2d[:, 0], stage1_prototypes_2d[:, 1], c=np.arange(num_classes), cmap='tab20', s=100)
for i in range(num_classes):
    plt.text(stage1_prototypes_2d[i, 0], stage1_prototypes_2d[i, 1], f'Class {i}', fontsize=8)
plt.title("After Stage 1 Separation")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# 第三张图：第二阶段原型分离
plt.subplot(1, 3, 3)
plt.scatter(stage2_prototypes_2d[:, 0], stage2_prototypes_2d[:, 1], c=np.arange(num_classes), cmap='tab20', s=100)
for i in range(num_classes):
    plt.text(stage2_prototypes_2d[i, 0], stage2_prototypes_2d[i, 1], f'Class {i}', fontsize=8)
plt.title("After Stage 2 Separation")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.tight_layout()
plt.savefig("prototype_separation_comparison.png")
plt.show()