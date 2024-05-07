from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import numpy as np
import random
import torch
class FashionMNIST:
    def __init__(self, clients, group_labels):
        # 获取数据集
        train_datasets = datasets.FashionMNIST(root='../data/', train=True,
                                               transform=transforms.ToTensor(), download=True)
        test_datasets = datasets.FashionMNIST(root='../data/', train=False,
                                              transform=transforms.ToTensor(), download=True)
        train_data = train_datasets.data
        self.raw_data = train_datasets.data
        self.train_labels = train_datasets.targets
        test_data = test_datasets.data
        self.test_labels = test_datasets.targets

        self.test_datasets = test_datasets
        # 归一化
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])
        train_data = train_data.float()
        self.train_data = np.multiply(train_data, 1.0 / 255.0)  # 数组对应元素位置相乘
        test_data = test_data.float()
        self.test_data = np.multiply(test_data, 1.0 / 255.0)
        test_size = 5
        self.water_data_base = np.multiply(train_data, 1.0/ 255.0)[:test_size]
        self.water_label_base = torch.tensor([10]*test_size)

        noise = 0.4*torch.randn(test_size, 784)
        self.water_data_noise = np.multiply(test_data, 1.0/ 255.0)[:test_size] + noise
        self.water_label_noise = torch.tensor([11]*test_size)
        self.water_data = torch.cat((self.water_data_base, self.water_data_noise), dim=0)
        self.water_label = torch.cat((self.water_label_base, self.water_label_noise), dim=0)

        self.train_data_size = train_data.shape[0]
        self.datasets = []
        # Data distribution
        label_ids = {label: np.where(self.train_labels == label)[0] for label in range(10)}
        group_ranges = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
        np.random.seed(123)
        random.seed(123)
        for client_id in range(0, 10):
            current_group_labels = [0,1,2,3,4,5,6,7,8,9]
            client_sample_size = np.random.randint(0, 12000)  # 随机选择200到3000之间的样本数量
            client_data_indices = np.concatenate([np.random.choice(label_ids[label], client_sample_size, replace=True) for label in current_group_labels])
            np.random.shuffle(client_data_indices)
            client_data_indices = client_data_indices[:client_sample_size]  # 确保样本数量不超过设定值
            client_data = self.train_data[client_data_indices]
            client_labels = self.train_labels[client_data_indices]
            self.datasets.append((client_data, client_labels))

        print("Data distribution process completed")

    def get_test_dataset(self):
        return self.test_data,self.test_labels

    def get_train_dataset(self):
        return self.datasets

    def get_water_dataset(self):
        return self.water_data, self.water_label
    def get_water_dataset(self):
        return self.water_data_noise, self.water_label_noise

if __name__ == '__main__':
    group_labels = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]  # Define label groups
    data = FashionMNIST(100, group_labels)
    train_datasets = data.get_train_dataset()
    print("end")