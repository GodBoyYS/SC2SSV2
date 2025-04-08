import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader2 import load_archived_hyperspectral_data, get_dataset_params
from train import train_prototypical
from test import test_prototypical
from Models.spectral_cnn1_2d import SpectralCNN, SpectralCNN2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(dataset_name="IP", apply_pca=False, n_components=50, k_shot=1):
    """
    主函数，用于训练和测试 SF2SS 模型。

    参数：
    - dataset_name: 数据集名称，"IP" 或 "SA"。
    - apply_pca: 是否应用 PCA 降维。
    - n_components: PCA 降维后的波段数。
    - k_shot: 每个类别的支持样本数（少样本设置）。
    """
    # 设置数据集划分比例（无验证集）
    if dataset_name == "IP":
        train_ratio = 0.1  # 10% 训练，90% 测试
    elif dataset_name == "SA":
        train_ratio = 0.1  # 10% 训练，90% 测试
    else:
        raise ValueError("dataset_name 必须是 'IP' 或 'SA'")

    # 加载数据集（val_ratio=0，不划分验证集）
    train_dataset, _, test_dataset = load_archived_hyperspectral_data(
        dataset_name=dataset_name,
        train_ratio=train_ratio,
        val_ratio=0.0,  # 不划分验证集
        min_train_samples=5,
        apply_pca=apply_pca,
        n_components=n_components
    )

    # 获取数据集参数
    params = get_dataset_params(dataset_name=dataset_name, apply_pca=apply_pca, n_components=n_components)

    # 初始化模型
    model_1d = SpectralCNN(
        input_size=params['input_size'],
        embedding_dim=64,
        num_classes=params['num_classes']
    ).to(device)
    model_2d = SpectralCNN2D(
        height=params['height'],
        width=params['width'],
        embedding_dim=64
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(model_1d.parameters()) + list(model_2d.parameters()),
        lr=0.001
    )

    # 设置保存路径
    save_path = f"./bestProto_{dataset_name}_{k_shot}shot.pth"

    # 训练
    model_1d, model_2d, save_path = train_prototypical(
        model_1d, model_2d,
        train_dataset=train_dataset,
        test_dataset=test_dataset,  # 仅使用测试集
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        params=params,
        num_epochs=100,
        n_way=5,
        k_shot=k_shot,
        q_query=5,
        save_path=save_path
    )

    # 测试
    test_prototypical(
        model_1d, model_2d,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device=device,
        save_path=save_path,
        params=params,
    )

if __name__ == "__main__":
    # 训练 IP 数据集（k_shot=1）
    # main(dataset_name="IP", apply_pca=False, n_components=50, k_shot=1)

    # 训练 SA 数据集（k_shot=1）
    # main(dataset_name="SA", apply_pca=False, n_components=50, k_shot=1)

    # 可选：训练 IP 数据集（k_shot=5）
    main(dataset_name="IP", apply_pca=False, n_components=50, k_shot=5)

    # 可选：训练 SA 数据集（k_shot=5）
    # main(dataset_name="SA", apply_pca=False, n_components=50, k_shot=5)