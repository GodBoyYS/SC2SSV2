import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

def to_2d_raw(data, height, width):
    """
    将 1D 光谱数据重塑为 2D 格式。
    参数：
    - data: 输入数据，形状为 (batch_size, 1, input_size)。
    - height: 2D 高度。
    - width: 2D 宽度。
    返回：
    - 重塑后的数据，形状为 (batch_size, 1, height, width)。
    """
    return data.view(-1, 1, height, width)

def compute_metrics(predictions, targets, num_classes):
    """
    计算 OA、AA 和 Kappa 系数。
    参数：
    - predictions: 预测标签列表。
    - targets: 真实标签列表。
    - num_classes: 类别数量。
    返回：
    - OA, AA, Kappa 系数。
    """
    # 计算混淆矩阵
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, target in zip(predictions, targets):
        conf_matrix[target, pred] += 1

    # 总体准确率 (OA)
    total_correct = np.trace(conf_matrix)  # 对角线元素之和
    total_samples = np.sum(conf_matrix)
    oa = total_correct / total_samples if total_samples > 0 else 0

    # 平均准确率 (AA)
    class_accuracy = []
    for i in range(num_classes):
        total_i = np.sum(conf_matrix[i, :])  # 真实类别 i 的总样本数
        if total_i > 0:
            acc_i = conf_matrix[i, i] / total_i  # 类别 i 的准确率
            class_accuracy.append(acc_i)
    aa = np.mean(class_accuracy) if class_accuracy else 0

    # Kappa 系数
    kappa = cohen_kappa_score(targets, predictions)

    return oa, aa, kappa

def evaluate_metrics(model_1d, model_2d, train_dataset, test_dataset, device, save_path, params, num_classes=16):
    """
    加载最佳模型参数并计算分类结果的 OA、AA 和 Kappa 系数。
    参数：
    - model_1d: 1D 模型 (SpectralCNN)。
    - model_2d: 2D 模型 (SpectralCNN2D)。
    - train_dataset: 训练数据集。
    - test_dataset: 测试数据集。
    - device: 设备 (CPU/GPU)。
    - save_path: 最佳模型参数路径。
    - params: 数据集参数 (包含 height, width 等)。
    - num_classes: 类别数量，默认为 16。
    """
    # 加载最佳模型参数
    checkpoint = torch.load(save_path, weights_only=True)
    model_1d.load_state_dict(checkpoint['model_1d'])
    model_2d.load_state_dict(checkpoint['model_2d'])
    model_1d.eval()
    model_2d.eval()

    # 获取 height 和 width
    height = params['height']
    width = params['width']

    # 权重参数
    emb_1d_weight = 0.75
    emb_2d_weight = 0.25

    # 存储所有预测和真实标签
    all_predictions = []
    all_targets = []

    # 每个类别的正确预测数和总数
    class_correct = {cls: 0 for cls in range(num_classes)}
    class_total = {cls: 0 for cls in range(num_classes)}

    with torch.no_grad():
        test_data = torch.tensor(test_dataset.data.numpy(), dtype=torch.float32).to(device).unsqueeze(1)
        test_labels = torch.tensor(test_dataset.labels.numpy(), dtype=torch.long).to(device)
        train_data = torch.tensor(train_dataset.data.numpy(), dtype=torch.float32).to(device).unsqueeze(1)
        train_labels = torch.tensor(train_dataset.labels.numpy(), dtype=torch.long).to(device)
        unique_labels = np.unique(train_labels.cpu().numpy())

        for query_idx in tqdm(range(test_data.size(0)), desc="Evaluating"):
            query_sample = test_data[query_idx:query_idx+1]
            query_label_true = test_labels[query_idx].item()

            # 构造支持集
            support_data, support_labels, support_class_labels = [], [], []
            for i, cls in enumerate(unique_labels):
                cls_indices = np.where(train_labels.cpu().numpy() == cls)[0]
                num_samples = min(5, len(cls_indices))  # 假设 k_shot=5
                if num_samples == 0:
                    continue
                perm = np.random.permutation(cls_indices)[:num_samples]
                support_data.append(train_data[perm].squeeze(1).cpu().numpy())
                support_labels.append(np.full(num_samples, i))
                support_class_labels.extend([cls] * num_samples)

            support_data = torch.tensor(np.concatenate(support_data), dtype=torch.float32).to(device).unsqueeze(1)
            support_labels = torch.tensor(np.concatenate(support_labels), dtype=torch.long).to(device)
            support_class_labels = torch.tensor(support_class_labels, dtype=torch.long).to(device)

            # 提取特征
            support_emb_1d = model_1d(support_data, support_class_labels)
            query_emb_1d = model_1d(query_sample, torch.tensor([query_label_true], dtype=torch.long).to(device))

            support_raw_2d = to_2d_raw(support_data, height, width)
            query_raw_2d = to_2d_raw(query_sample, height, width)
            support_emb_2d = model_2d(support_raw_2d)
            query_emb_2d = model_2d(query_raw_2d)

            # 融合特征
            support_emb = emb_1d_weight * support_emb_1d + emb_2d_weight * support_emb_2d
            query_emb = emb_1d_weight * query_emb_1d + emb_2d_weight * query_emb_2d

            # 计算原型
            prototypes = []
            for i in range(len(unique_labels)):
                cls_embeddings = support_emb[support_labels == i]
                prototype = cls_embeddings.mean(dim=0)
                prototypes.append(prototype)
            prototypes = torch.stack(prototypes)

            # 预测
            distances = torch.cdist(query_emb, prototypes)
            pred_idx = distances.argmin(dim=1).item()
            pred_label = unique_labels[pred_idx]

            # 存储预测和真实标签
            all_predictions.append(pred_label)
            all_targets.append(query_label_true)

            # 更新每个类别的统计
            cls = query_label_true
            class_total[cls] += 1
            if pred_label == cls:
                class_correct[cls] += 1

    # 计算 OA、AA 和 Kappa 系数
    oa, aa, kappa = compute_metrics(all_predictions, all_targets, num_classes)

    # 打印结果
    print(f"\nEvaluation Results:")
    print(f"Overall Accuracy (OA): {oa*100:.2f}%")
    print(f"Average Accuracy (AA): {aa*100:.2f}%")
    print(f"Kappa Coefficient: {kappa:.4f}")

    # 打印每个类别的准确率
    print("\nAccuracy per Class:")
    for cls in range(num_classes):
        if class_total[cls] > 0:
            acc = class_correct[cls] / class_total[cls] * 100
            print(f"Class {cls+1} (mapped as {cls}): Accuracy = {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")
        else:
            print(f"Class {cls+1} (mapped as {cls}): No samples tested")

    return oa, aa, kappa

if __name__ == "__main__":
    import sys
    sys.path.append('/')
    from data_loader import load_archived_hyperspectral_data, get_dataset_params
    from Models.spectral_cnn1_2d import SpectralCNN, SpectralCNN2D

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集和参数
    dataset_name = "SA"  # 可改为 "SA"
    train_ratio = 0.1  # IP 数据集 10% 训练
    k_shot = 1  # 与 main2.py 中的调用一致

    # 加载数据集
    train_dataset, _, test_dataset = load_archived_hyperspectral_data(
        dataset_name=dataset_name,
        train_ratio=train_ratio,
        val_ratio=0.0,
        min_train_samples=5,
        apply_pca=False,
        n_components=50
    )

    # 获取数据集参数
    params = get_dataset_params(dataset_name=dataset_name, apply_pca=False, n_components=50)

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

    # 模型保存路径
    save_path = f"./bestProto_{dataset_name}_{k_shot}shot.pth"

    # 评估模型
    oa, aa, kappa = evaluate_metrics(
        model_1d, model_2d,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device=device,
        save_path=save_path,
        params=params,
        num_classes=params['num_classes']
    )