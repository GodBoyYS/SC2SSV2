import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader
import os

class ArchivedHyperspectralDataset(Dataset):
    def __init__(self, archived_data, archived_labels):
        self.data = torch.tensor(archived_data, dtype=torch.float32)
        self.labels = torch.tensor(archived_labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_archived_hyperspectral_data(dataset_name="IP", train_ratio=0.1, val_ratio=0.0, min_train_samples=5, apply_pca=False, n_components=50):
    """
    加载高光谱数据集（支持 IP 和 SA 数据集）。

    参数：
    - dataset_name: 数据集名称，"IP" 或 "SA"。
    - train_ratio: 训练集比例。
    - val_ratio: 验证集比例（若为 0，则不划分验证集）。
    - min_train_samples: 每个类别的最小训练样本数。
    - apply_pca: 是否应用 PCA 降维。
    - n_components: PCA 降维后的波段数。

    返回：
    - train_dataset, val_dataset, test_dataset: 训练、验证和测试数据集（若 val_ratio=0，则 val_dataset 为空）。
    """
    # 根据 dataset_name 选择数据路径
    if dataset_name == "IP":
        image_path = './datasets/Indian_pines_corrected.mat'
        label_path = './datasets/Indian_pines_gt.mat'
        image_key = 'indian_pines_corrected'
        label_key = 'indian_pines_gt'
    elif dataset_name == "SA":
        image_path = './datasets/Salinas_corrected.mat'
        label_path = './datasets/Salinas_gt.mat'
        image_key = 'salinas_corrected'
        label_key = 'salinas_gt'
    else:
        raise ValueError("dataset_name 必须是 'IP' 或 'SA'")

    # 加载数据
    if not os.path.exists(image_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"数据文件未找到：{image_path} 或 {label_path}")

    image_data = loadmat(image_path)[image_key]  # (H, W, B)
    labels = loadmat(label_path)[label_key]      # (H, W)

    # 数据预处理
    image_data = image_data.astype(np.float32)
    scaler = StandardScaler()
    image_data_flat = np.reshape(image_data, (-1, image_data.shape[2]))  # (H*W, B)
    image_data_flat = scaler.fit_transform(image_data_flat)

    # 应用 PCA 降维（可选）
    if apply_pca:
        pca = PCA(n_components=n_components)
        image_data_flat = pca.fit_transform(image_data_flat)
        print(f"数据集 {dataset_name} 的 PCA 降维后解释方差比例: {sum(pca.explained_variance_ratio_):.4f}")
    else:
        n_components = image_data.shape[2]  # 使用原始波段数

    labels_flat = labels.reshape(-1)  # (H*W,)

    # 按类别归档
    unique_labels = np.unique(labels_flat)
    class_archives = {}
    for cls in unique_labels:
        cls_indices = np.where(labels_flat == cls)[0]
        cls_data = image_data_flat[cls_indices]
        class_archives[cls] = {'data': cls_data, 'num_samples': len(cls_indices)}

    # 删除背景类别（标签为 0）
    if 0 in class_archives:
        del class_archives[0]

    # 合并数据和标签
    archived_data = np.concatenate([class_archives[cls]['data'] for cls in class_archives], axis=0)
    archived_labels = np.concatenate([np.full(class_archives[cls]['num_samples'], cls) for cls in class_archives], axis=0)
    archived_labels = archived_labels - 1  # 映射为 0-(num_classes-1)

    # 按类别划分训练、验证和测试集
    train_data_list, val_data_list, test_data_list = [], [], []
    train_labels_list, val_labels_list, test_labels_list = [], [], []
    for cls in range(len(class_archives)):
        cls_indices = np.where(archived_labels == cls)[0]
        cls_data = archived_data[cls_indices]
        cls_labels = archived_labels[cls_indices]
        total_samples = len(cls_data)

        if total_samples <= min_train_samples:
            train_data_list.append(cls_data)
            train_labels_list.append(cls_labels)
        else:
            perm = np.random.permutation(total_samples)
            train_size = max(min_train_samples, int(total_samples * train_ratio))
            val_size = int(total_samples * val_ratio) if val_ratio > 0 else 0

            # 划分训练集
            train_data_list.append(cls_data[perm[:train_size]])
            train_labels_list.append(cls_labels[perm[:train_size]])

            # 划分验证集和测试集
            remaining_data = cls_data[perm[train_size:]]
            remaining_labels = cls_labels[perm[train_size:]]
            if val_ratio > 0 and len(remaining_data) > val_size:
                val_data_list.append(remaining_data[:val_size])
                val_labels_list.append(remaining_labels[:val_size])
                test_data_list.append(remaining_data[val_size:])
                test_labels_list.append(remaining_labels[val_size:])
            else:
                test_data_list.append(remaining_data)
                test_labels_list.append(remaining_labels)

    # 合并数据
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    val_data = np.concatenate(val_data_list, axis=0) if val_data_list else np.array([])
    val_labels = np.concatenate(val_labels_list, axis=0) if val_labels_list else np.array([])
    test_data = np.concatenate(test_data_list, axis=0) if test_data_list else np.array([])
    test_labels = np.concatenate(test_labels_list, axis=0) if test_labels_list else np.array([])

    # 创建 Dataset 实例
    train_dataset = ArchivedHyperspectralDataset(train_data, train_labels)
    val_dataset = ArchivedHyperspectralDataset(val_data, val_labels) if val_data.size > 0 else None
    test_dataset = ArchivedHyperspectralDataset(test_data, test_labels)

    print(f"数据集 {dataset_name} 的训练集样本数: {len(train_dataset)}")
    if val_dataset:
        print(f"数据集 {dataset_name} 的验证集样本数: {len(val_dataset)}")
    print(f"数据集 {dataset_name} 的测试集样本数: {len(test_dataset)}")
    print(f"数据集 {dataset_name} 的类别数: {len(class_archives)}, 类别: {list(class_archives.keys())}")

    return train_dataset, val_dataset, test_dataset

def get_dataset_params(dataset_name="IP", apply_pca=False, n_components=50):
    """
    获取数据集参数，用于后续训练。

    参数：
    - dataset_name: 数据集名称，"IP" 或 "SA"。
    - apply_pca: 是否应用 PCA 降维。
    - n_components: PCA 降维后的波段数。

    返回：
    - params: 包含 input_size, height, width, num_classes 的字典。
    """
    if dataset_name == "IP":
        num_classes = 16
        original_bands = 200  # IP 数据集原始波段数
    elif dataset_name == "SA":
        num_classes = 16
        original_bands = 204  # SA 数据集原始波段数
    else:
        raise ValueError("dataset_name 必须是 'IP' 或 'SA'")

    # 确定 input_size
    input_size = n_components if apply_pca else original_bands

    # 确定 height 和 width，使得 height * width = input_size
    sqrt_input = int(np.sqrt(input_size))
    for height in range(sqrt_input, 0, -1):
        if input_size % height == 0:
            width = input_size // height
            break
    else:
        height = input_size
        width = 1

    return {
        'input_size': input_size,  # 光谱带数（降维后或原始）
        'height': height,          # 2D 折叠高度
        'width': width,            # 2D 折叠宽度
        'num_classes': num_classes # 类别数
    }

def test_data_loading():
    """
    测试数据加载并打印数据集信息。
    """
    apply_pca = False
    n_components = 50

    # 加载 IP 数据集（10% 训练，90% 测试，无验证集）
    ip_train_dataset, ip_val_dataset, ip_test_dataset = load_archived_hyperspectral_data(
        dataset_name="IP", train_ratio=0.1, val_ratio=0.0, min_train_samples=5, apply_pca=apply_pca, n_components=n_components
    )

    # 加载 SA 数据集（5% 训练，95% 测试，无验证集）
    sa_train_dataset, sa_val_dataset, sa_test_dataset = load_archived_hyperspectral_data(
        dataset_name="SA", train_ratio=0.05, val_ratio=0.0, min_train_samples=5, apply_pca=apply_pca, n_components=n_components
    )

    ip_params = get_dataset_params(dataset_name="IP", apply_pca=apply_pca, n_components=n_components)
    sa_params = get_dataset_params(dataset_name="SA", apply_pca=apply_pca, n_components=n_components)

    # 创建 DataLoader
    ip_train_loader = DataLoader(ip_train_dataset, batch_size=16, shuffle=True)
    ip_test_loader = DataLoader(ip_test_dataset, batch_size=16, shuffle=False)

    sa_train_loader = DataLoader(sa_train_dataset, batch_size=16, shuffle=True)
    sa_test_loader = DataLoader(sa_test_dataset, batch_size=16, shuffle=False)

    # 打印 IP 数据集信息
    print("\n=== IP 数据集信息 ===")
    print(f"训练集样本数: {len(ip_train_dataset)}")
    print(f"测试集样本数: {len(ip_test_dataset)}")
    print(f"类别数: {len(np.unique([label.item() for _, label in ip_train_dataset]))}")
    print(f"类别: {np.unique([label.item() for _, label in ip_train_dataset])}")
    print(f"参数: {ip_params}")

    # 打印 SA 数据集信息
    print("\n=== SA 数据集信息 ===")
    print(f"训练集样本数: {len(sa_train_dataset)}")
    print(f"测试集样本数: {len(sa_test_dataset)}")
    print(f"类别数: {len(np.unique([label.item() for _, label in sa_train_dataset]))}")
    print(f"类别: {np.unique([label.item() for _, label in sa_train_dataset])}")
    print(f"参数: {sa_params}")

    # 测试 DataLoader
    print("\n=== 测试 DataLoader ===")
    for data, labels in ip_train_loader:
        print(f"IP 训练集批次 - 数据形状: {data.shape}, 标签形状: {labels.shape}")
        break
    for data, labels in sa_train_loader:
        print(f"SA 训练集批次 - 数据形状: {data.shape}, 标签形状: {labels.shape}")
        break

if __name__ == "__main__":
    test_data_loading()