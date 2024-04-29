from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import numpy as np
import random
g_train_data = './108resnetv2_train.csv'
g_test_data = './108resnetv2_test.csv'
g_val_data = './108resnetv2_val.csv'
import pandas as pd
class GetDataSet(object):
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self.val_data = None
        self.val_label = None
        self.val_data_size = None
        self._index_in_train_epoch = 0

    def load_data(self):
        data = pd.read_csv(g_train_data)
        train_x = np.array(data.iloc[:, 1:].values)
        train_y = np.array(data['species'].values)

        data_test = pd.read_csv(g_test_data)
        test_x = np.array(data_test.iloc[:, 1:].values)
        test_y = np.array(data_test['species'].values)

        data_val = pd.read_csv(g_val_data)
        val_x = np.array(data_val.iloc[:, 1:].values)
        val_y = np.array(data_val['species'].values)

        self.train_data_size = train_x.shape[0]
        self.test_data_size = test_x.shape[0]
        self.val_data_size = val_x.shape[0]

        self.train_data = train_x
        self.train_label = train_y
        self.test_data = test_x
        self.test_label = test_y
        self.val_data = val_x
        self.val_label = val_y

class FashionMNIST:
    def __init__(self):
        group_labels = [list(range(18)), list(range(18, 36)), list(range(36, 54)), list(range(54, 72)), list(range(72, 90)),
         list(range(90, 108))]
        # 获取数据集
        train_datasets = GetDataSet()
        train_datasets.load_data()
        self.train_data = train_datasets.train_data
        self.train_labels = train_datasets.train_label
        self.datasets = []
        # Data distribution
        label_ids = {label: np.where(train_datasets.train_label == label)[0] for label in range(108)}
        group_ranges = [(0, 78),(78,84), (84, 90), (90, 96), (96, 102), (102, 108)]
        np.random.seed(123)
        random.seed(123)
        # group_ranges = [(0, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
        for group_id, (start, end) in enumerate(group_ranges):
            current_group_labels = group_labels[group_id]
            for client_id in range(start, end):
                client_sample_size = np.random.randint(30, 384)  # 随机选择200到3000之间的样本数量
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

if __name__ == '__main__':
    # group_labels = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]  # Define label groups
    group_labels = [list(range(18)), list(range(18, 36)), list(range(36, 54)), list(range(54, 72)), list(range(72, 90)), list(range(90, 108))]
  # Define label groups
    data = FashionMNIST(100, group_labels)
    train_datasets = data.get_train_dataset()
    print("end")