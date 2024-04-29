import numpy as np
from torchvision import datasets, transforms
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
# 定义转换
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# 加载Fashion-MNIST训练数据集
fashion_mnist_dataset = datasets.FashionMNIST(root='./FedAvg/data/MNIST/', train=True, download=True, transform=transform)


# 函数：根据客户端组别分配Fashion-MNIST数据集
def allocate_data_to_clients(dataset, num_clients=100, num_labels_per_group=2):
    # 每个标签的样本索引
    label_to_indices = {label: np.where(dataset.targets == label)[0] for label in range(10)}

    # 客户端数据分配字典
    client_data_indices = defaultdict(list)

    # 客户端分组
    client_groups = {
        1: range(1, 61),
        2: range(61, 71),
        3: range(71, 81),
        4: range(81, 91),
        5: range(91, 101)
    }

    # 分配数据
    for group, clients in client_groups.items():
        # 每组分配两个标签
        labels = [((group - 1) * 2) % 10, ((group - 1) * 2 + 1) % 10]
        for client in clients:
            # 模拟不平衡的样本分配
            # 假设每个客户端至少有100个样本，然后随机增加
            num_samples_label_1 = 100 + np.random.randint(0, 5900)
            num_samples_label_2 = 100 + np.random.randint(0, 5900)
            client_data_indices[client].extend(
                np.random.choice(label_to_indices[labels[0]], num_samples_label_1, replace=False).tolist())
            client_data_indices[client].extend(
                np.random.choice(label_to_indices[labels[1]], num_samples_label_2, replace=False).tolist())

    return client_data_indices


# 进行数据分配
client_data_indices = allocate_data_to_clients(fashion_mnist_dataset)

def create_client_dataloaders(dataset, client_data_indices, batch_size=32):
    client_dataloaders = {}
    for client_id, indices in client_data_indices.items():
        # 创建数据子集
        subset = Subset(dataset, indices)
        # 创建数据加载器
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_dataloaders[client_id] = dataloader
    return client_dataloaders
# 创建客户端数据加载器
client_dataloaders = create_client_dataloaders(fashion_mnist_dataset, client_data_indices)
print(client_dataloaders[1])