import torch
import numpy as np
from tqdm import tqdm

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

def test_prototypical(model_1d, model_2d, train_dataset, test_dataset, device, save_path, params):
    emb_1d_weight = 0.75
    emb_2d_weight = 0.25
    checkpoint = torch.load(save_path, weights_only=True)  # 修改：设置 weights_only=True
    model_1d.load_state_dict(checkpoint['model_1d'])
    model_2d.load_state_dict(checkpoint['model_2d'])
    model_1d.eval()
    model_2d.eval()

    # 获取 height 和 width
    height = params['height']
    width = params['width']

    final_test_correct = 0
    final_test_total = 0
    class_correct = {cls: 0 for cls in range(16)}
    class_total = {cls: 0 for cls in range(16)}

    with torch.no_grad():
        test_data = torch.tensor(test_dataset.data.numpy(), dtype=torch.float32).to(device).unsqueeze(1)
        test_labels = torch.tensor(test_dataset.labels.numpy(), dtype=torch.long).to(device)
        train_data = torch.tensor(train_dataset.data.numpy(), dtype=torch.float32).to(device).unsqueeze(1)
        train_labels = torch.tensor(train_dataset.labels.numpy(), dtype=torch.long).to(device)
        unique_labels = np.unique(train_labels.cpu().numpy())

        for query_idx in tqdm(range(test_data.size(0)), desc="Final Test"):
            query_sample = test_data[query_idx:query_idx+1]
            query_label_true = test_labels[query_idx].item()

            support_data, support_labels, support_class_labels = [], [], []
            for i, cls in enumerate(unique_labels):
                cls_indices = np.where(train_labels.cpu().numpy() == cls)[0]
                num_samples = min(5, len(cls_indices))  # 假设 k_shot=5，需根据实际调用调整
                if num_samples == 0:
                    continue
                perm = np.random.permutation(cls_indices)[:num_samples]
                support_data.append(train_data[perm].squeeze(1).cpu().numpy())
                support_labels.append(np.full(num_samples, i))
                support_class_labels.extend([cls] * num_samples)

            support_data = torch.tensor(np.concatenate(support_data), dtype=torch.float32).to(device).unsqueeze(1)
            support_labels = torch.tensor(np.concatenate(support_labels), dtype=torch.long).to(device)
            support_class_labels = torch.tensor(support_class_labels, dtype=torch.long).to(device)

            support_emb_1d = model_1d(support_data, support_class_labels)
            query_emb_1d = model_1d(query_sample, torch.tensor([query_label_true], dtype=torch.long).to(device))

            support_raw_2d = to_2d_raw(support_data, height, width)  # 修改：传递 height 和 width
            query_raw_2d = to_2d_raw(query_sample, height, width)    # 修改：传递 height 和 width
            support_emb_2d = model_2d(support_raw_2d)
            query_emb_2d = model_2d(query_raw_2d)

            support_emb = emb_1d_weight * support_emb_1d + emb_2d_weight * support_emb_2d
            query_emb = emb_1d_weight * query_emb_1d + emb_2d_weight * query_emb_2d

            prototypes = []
            for i in range(len(unique_labels)):
                cls_embeddings = support_emb[support_labels == i]
                prototype = cls_embeddings.mean(dim=0)
                prototypes.append(prototype)
            prototypes = torch.stack(prototypes)

            distances = torch.cdist(query_emb, prototypes)
            pred_idx = distances.argmin(dim=1).item()
            pred_label = unique_labels[pred_idx]

            final_test_total += 1
            cls = query_label_true
            class_total[cls] += 1
            if pred_label == cls:
                final_test_correct += 1
                class_correct[cls] += 1

    final_test_accuracy = final_test_correct / final_test_total if final_test_total > 0 else 0
    print(f"\nFinal Test Accuracy with Best Model: {final_test_accuracy*100:.2f}% (Tested {final_test_total}/{test_data.size(0)} samples)")

    print("\nFinal Test Accuracy per Class with Best Model:")
    for cls in range(16):
        if class_total[cls] > 0:
            acc = class_correct[cls] / class_total[cls] * 100
            print(f"Class {cls+1} (mapped as {cls}): Accuracy = {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")
        else:
            print(f"Class {cls+1} (mapped as {cls}): No samples tested")