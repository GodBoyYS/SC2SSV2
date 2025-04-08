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

def prototype_separation_loss(prototypes, target_pairs, selected_classes, margin=1.0):
    loss = 0.0
    num_pairs = 0
    selected_classes_set = set(selected_classes.tolist())
    for cls_i, cls_j in target_pairs:
        if cls_i in selected_classes_set and cls_j in selected_classes_set:
            idx_i = np.where(selected_classes == cls_i)[0][0]
            idx_j = np.where(selected_classes == cls_j)[0][0]
            proto_i = prototypes[idx_i]
            proto_j = prototypes[idx_j]
            dist = torch.cdist(proto_i.unsqueeze(0), proto_j.unsqueeze(0))
            loss += torch.relu(margin - dist)
            num_pairs += 1
    return loss / num_pairs if num_pairs > 0 else 0.0

def compute_confusion_matrix(predictions, targets, num_classes=16):
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for pred, target in zip(predictions, targets):
        conf_matrix[target, pred] += 1
    return conf_matrix

def get_confused_pairs(conf_matrix, threshold=0.05):
    num_classes = conf_matrix.size(0)
    confused_pairs = []
    for i in range(num_classes):
        total_i = conf_matrix[i].sum().item()
        if total_i == 0:
            continue
        for j in range(num_classes):
            if i != j and conf_matrix[i, j].item() / total_i > threshold:
                confused_pairs.append((i, j))
    return confused_pairs

def get_small_sample_pairs(train_labels, sample_threshold=50):
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    small_sample_classes = [label for label, count in zip(unique_labels, counts) if count < sample_threshold]
    small_sample_pairs = []
    for cls_i in small_sample_classes:
        for cls_j in range(16):
            if cls_i != cls_j:
                small_sample_pairs.append((cls_i, cls_j))
    return small_sample_pairs

def train_prototypical(model_1d, model_2d, train_dataset, test_dataset, criterion, optimizer, device, params, num_epochs=20, n_way=5, k_shot=5, q_query=5, save_path='./bestProto.pth'):
    emb_1d_weight = 0.75
    emb_2d_weight = 0.25
    model_1d.to(device)
    model_2d.to(device)
    best_test_accuracy = 0.0
    confused_pairs = []
    small_sample_pairs = get_small_sample_pairs(train_dataset.labels.numpy(), sample_threshold=50)

    # 获取 height 和 width
    height = params['height']
    width = params['width']

    for epoch in range(num_epochs):
        model_1d.train()
        model_2d.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        with tqdm(range(100), unit="batch", desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as tepoch:
            for _ in tepoch:
                train_data = train_dataset.data.numpy()
                train_labels = train_dataset.labels.numpy()
                unique_labels = np.unique(train_labels)
                if len(unique_labels) < n_way:
                    continue
                selected_classes = np.random.choice(unique_labels, n_way, replace=False)

                support_data, support_labels = [], []
                query_data, query_labels = [], []
                support_class_labels, query_class_labels = [], []
                query_targets_list = []
                for i, cls in enumerate(selected_classes):
                    cls_indices = np.where(train_labels == cls)[0]
                    num_support = min(k_shot, len(cls_indices))
                    num_query = min(q_query, max(0, len(cls_indices) - num_support))
                    if num_support == 0 or num_query == 0:
                        continue
                    perm = np.random.permutation(cls_indices)
                    support_data.append(train_data[perm[:num_support]])
                    support_labels.append(np.full(num_support, i))
                    support_class_labels.extend([cls] * num_support)
                    query_data.append(train_data[perm[num_support:num_support + num_query]])
                    query_labels.append(np.full(num_query, i))
                    query_class_labels.extend([cls] * num_query)
                    query_targets_list.extend([i] * num_query)

                if len(support_data) != n_way or len(query_data) != n_way:
                    continue

                support_data = torch.tensor(np.concatenate(support_data), dtype=torch.float32).to(device).unsqueeze(1)
                support_labels = torch.tensor(np.concatenate(support_labels), dtype=torch.long).to(device)
                support_class_labels = torch.tensor(support_class_labels, dtype=torch.long).to(device)
                query_data = torch.tensor(np.concatenate(query_data), dtype=torch.float32).to(device).unsqueeze(1)
                query_labels = torch.tensor(np.concatenate(query_labels), dtype=torch.long).to(device)
                query_class_labels = torch.tensor(query_class_labels, dtype=torch.long).to(device)
                query_targets = torch.tensor(query_targets_list, dtype=torch.long).to(device)

                support_emb_1d = model_1d(support_data, support_class_labels)
                query_emb_1d = model_1d(query_data, query_class_labels)
                support_raw_2d = to_2d_raw(support_data, height, width)  # 修改：传递 height 和 width
                query_raw_2d = to_2d_raw(query_data, height, width)      # 修改：传递 height 和 width
                support_emb_2d = model_2d(support_raw_2d)
                query_emb_2d = model_2d(query_raw_2d)

                support_emb = emb_1d_weight * support_emb_1d + emb_2d_weight * support_emb_2d
                query_emb = emb_1d_weight * query_emb_1d + emb_2d_weight * query_emb_2d

                prototypes = []
                for i in range(n_way):
                    cls_embeddings = support_emb[support_labels == i]
                    prototype = cls_embeddings.mean(dim=0)
                    prototypes.append(prototype)
                prototypes = torch.stack(prototypes)

                distances = torch.cdist(query_emb, prototypes)
                loss_1d = criterion(-torch.cdist(query_emb_1d, prototypes), query_targets)
                loss_2d = criterion(-torch.cdist(query_emb_2d, prototypes), query_targets)
                loss = 0.75 * loss_1d + 0.25 * loss_2d

                if confused_pairs:
                    sep_loss_confused = prototype_separation_loss(prototypes, confused_pairs, selected_classes, margin=5.0)
                else:
                    sep_loss_confused = 0.0

                if small_sample_pairs:
                    sep_loss_small = prototype_separation_loss(prototypes, small_sample_pairs, selected_classes, margin=5.0)
                else:
                    sep_loss_small = 0.0

                total_loss = 0.4 * loss + 0.3 * sep_loss_confused + 0.3 * sep_loss_small

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item()
                _, predicted = torch.min(distances, 1)
                train_correct += (predicted == query_targets).sum().item()
                train_total += query_targets.size(0)

                tepoch.set_postfix(loss=train_loss / (tepoch.n + 1), accuracy=train_correct / train_total)

        train_accuracy = train_correct / train_total if train_total > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/100:.4f}, Train Accuracy: {train_accuracy*100:.2f}%")

        # 验证阶段（这里实际上是测试阶段，原始代码中没有验证集）
        model_1d.eval()
        model_2d.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for _ in range(50):
                test_data = test_dataset.data.numpy()
                test_labels = test_dataset.labels.numpy()
                unique_labels = np.unique(test_labels)
                if len(unique_labels) < n_way:
                    continue
                selected_classes = np.random.choice(unique_labels, n_way, replace=False)

                support_data, support_labels = [], []
                query_data, query_labels = [], []
                support_class_labels, query_class_labels = [], []
                query_targets_list = []
                for i, cls in enumerate(selected_classes):
                    cls_indices = np.where(test_labels == cls)[0]
                    num_support = min(k_shot, len(cls_indices))
                    num_query = min(q_query, max(0, len(cls_indices) - num_support))
                    if num_support == 0 or num_query == 0:
                        continue
                    perm = np.random.permutation(cls_indices)
                    support_data.append(test_data[perm[:num_support]])
                    support_labels.append(np.full(num_support, i))
                    support_class_labels.extend([cls] * num_support)
                    query_data.append(test_data[perm[num_support:num_support + num_query]])
                    query_labels.append(np.full(num_query, i))
                    query_class_labels.extend([cls] * num_query)
                    query_targets_list.extend([i] * num_query)

                if len(support_data) != n_way or len(query_data) != n_way:
                    continue

                support_data = torch.tensor(np.concatenate(support_data), dtype=torch.float32).to(device).unsqueeze(1)
                support_labels = torch.tensor(np.concatenate(support_labels), dtype=torch.long).to(device)
                support_class_labels = torch.tensor(support_class_labels, dtype=torch.long).to(device)
                query_data = torch.tensor(np.concatenate(query_data), dtype=torch.float32).to(device).unsqueeze(1)
                query_labels = torch.tensor(np.concatenate(query_labels), dtype=torch.long).to(device)
                query_class_labels = torch.tensor(query_class_labels, dtype=torch.long).to(device)
                query_targets = torch.tensor(query_targets_list, dtype=torch.long).to(device)

                support_emb_1d = model_1d(support_data, support_class_labels)
                query_emb_1d = model_1d(query_data, query_class_labels)
                support_raw_2d = to_2d_raw(support_data, height, width)  # 修改：传递 height 和 width
                query_raw_2d = to_2d_raw(query_data, height, width)      # 修改：传递 height 和 width
                support_emb_2d = model_2d(support_raw_2d)
                query_emb_2d = model_2d(query_raw_2d)

                support_emb = emb_1d_weight * support_emb_1d + emb_2d_weight * support_emb_2d
                query_emb = emb_1d_weight * query_emb_1d + emb_2d_weight * query_emb_2d

                prototypes = []
                for i in range(n_way):
                    cls_embeddings = support_emb[support_labels == i]
                    prototype = cls_embeddings.mean(dim=0)
                    prototypes.append(prototype)
                prototypes = torch.stack(prototypes)

                distances = torch.cdist(query_emb, prototypes)
                loss = criterion(-distances, query_targets)

                test_loss += loss.item()
                _, predicted = torch.min(distances, 1)
                test_correct += (predicted == query_targets).sum().item()
                test_total += query_targets.size(0)

                all_predictions.extend(predicted.cpu().tolist())
                all_targets.extend(query_targets.cpu().tolist())

        test_accuracy = test_correct / test_total if test_total > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss/50:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")

        conf_matrix = compute_confusion_matrix(all_predictions, all_targets, num_classes=n_way)
        confused_pairs = get_confused_pairs(conf_matrix, threshold=0.05)
        if confused_pairs or small_sample_pairs:
            print(f"Epoch [{epoch+1}/{num_epochs}], Confused Pairs: {confused_pairs}, Small Sample Pairs: {small_sample_pairs}")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save({'model_1d': model_1d.state_dict(), 'model_2d': model_2d.state_dict()}, save_path)
            print(f"Best model saved at epoch {epoch+1} with Test Accuracy: {best_test_accuracy*100:.2f}%")

    return model_1d, model_2d, save_path