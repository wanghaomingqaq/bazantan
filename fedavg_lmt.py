"""
只同步一次参数
"""
import sys
from ddpg import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
import random
import copy
from torch.nn import Linear
from torch import FloatTensor
import pandas as pd
from ddpg import updateDDPG
import random
from lmt import GetDataSet,FashionMNIST
g_exporStep = 1  # 第一步随机探索
g_num_steps = 4,  # 进行学习的频次
g_memory_size = 500,  # 经验回放池的容量
g_replay_start_size = 100,  # 开始回放的次数
g_update_target_steps = 100,  # 同步参数的次数
g_gamma = 0.9,
epoch = 1
batchsize = 8
learning_rate = 0.0001
num_comm = 10000
num_of_one_epoch = 200
num_of_client = 108
num_in_comm = 108  # 每次抽取个数
# g_train_data = r'D:\github\workspace\pytorch_study\dataset\108resnetv2_train.csv'
# g_test_data = r'D:\github\workspace\pytorch_study\dataset\108resnetv2_test.csv'
# g_val_data = r'D:\github\workspace\pytorch_study\dataset\108resnetv2_val.csv'
# g_train_data = './108resnetv2_train.csv'
# g_test_data = './108resnetv2_test.csv'
# g_val_data = './108resnetv2_val.csv'

class LinearNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = self.__mlp()

    def __mlp(self):
        return torch.nn.Sequential(
            Linear(4096, 1024),
            torch.nn.ReLU(),
            Linear(1024, 512),
            torch.nn.ReLU(),
            Linear(512, 128),
            torch.nn.ReLU(),
            Linear(128, 108)
        )

    def forward(self, x):
        return self.mlp(x)
# data

def loss_ext(train_dl, net):
    l = 0
    i = 0
    with torch.no_grad():
        for data, label in train_dl:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            loss_f = torch.nn.CrossEntropyLoss()
            loss = loss_f(preds, label)
            l += loss.item()
            i += 1
    l /= i
    return l
# clients
class client(object):
    def __init__(self, train_x, train_y, dev):
        self.train_x = train_x
        self.train_y = train_y
        self.dev = dev
        self.train_ds = None
        self.train_dl = None
        self.local_parameters = None
        self.la = 0
        self.lb = 0

    def localUpdate(self, localEpoch, localBatchSize, Net, loss_function, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        # 载入Client自有数据集
        # 加载本地数据
        self.train_x = self.train_x.astype(np.float32)
        self.train_y = self.train_y.astype(np.int64)
        self.train_ds = TensorDataset(torch.tensor(self.train_x), torch.tensor(self.train_y))

        self.train_dl = DataLoader((self.train_ds), batch_size=localBatchSize, shuffle=True)
        # 设置迭代次数
        self.lb = loss_ext(self.train_dl, Net)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = Net(data)
                # print(preds.shape)
                loss = loss_function(preds, label)
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        Net.state_dict()
        self.la = loss_ext(self.train_dl, Net)
        k_n = len(self.train_dl)
        return Net.state_dict(), self.lb, self.la, k_n

class ClientsGroup(object):
    def __init__(self, numOfClients, dev,data_dis='imbalance'):
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.val_data_loader = None
        self.data_dis = data_dis
        self.dataSetBalanceAllocation()
        self.data = []
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        group_labels = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]  # Define label groups
        data = FashionMNIST()
        train_datasets = data.get_train_dataset()
        for i in range(num_of_client):
            someone = client(train_datasets[i][0], train_datasets[i][1], self.dev)
            #
            # print(someone.train_x.shape)
            self.clients_set['client{}'.format(i)] = someone

def get_reward(l_a,l_b):
    l_b_avg = sum(l_b) / len(l_b)
    l_a_avg = sum(l_a) / len(l_a)
    l_a_diff = max(l_a) - min(l_a)
    l_b_diff = max(l_b) - min(l_b)
    r_t = l_b_avg + l_a_avg + l_a_diff + l_b_diff
    return r_t

def acc(com):
    sum_accu = 0
    num = 0
    for data, label in testDataLoader:
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        preds = torch.argmax(preds, dim=1)
        sum_accu += (preds == label).float().mean()
        num += 1
    print("\n" + 'accuracy: {}'.format(sum_accu / num))
    print("\n" + 'comm: {}'.format(com))
    sys.stdout.flush()

def env_geneWk(global_parameters,comn):
    # start_time = time.time()
    clients_in_comm = ['client{}'.format(i) for i in range(num_in_comm)]
    sum_parameters = None

    for idx, client in enumerate(clients_in_comm):
        # 本地更新
        local_parameters, loss_b, loss_a, k_n = myClients.clients_set[client].localUpdate(
            epoch, batchsize, net, loss_function, optimizer, global_parameters)

        # 使用动作 a_t 权重更新参数
        weight = 1/ num_of_client
        weighted_parameters = {key: weight * var.clone() for key, var in local_parameters.items()}

        if sum_parameters is None:
            sum_parameters = weighted_parameters
        else:
            for var in sum_parameters:
                sum_parameters[var] += weighted_parameters[var]
    acc(comn)
    for var in sum_parameters:
        sum_parameters[var] = sum_parameters[var]

    for var in global_parameters:
        global_parameters[var] = sum_parameters[var]
    return global_parameters
def act(self, state):
    if np.random.rand() < self.epsilon:
        return np.random.randint(0, 2)
    with torch.no_grad():
        q_values = self.q_network(torch.FloatTensor(state))
    return torch.argmax(q_values).item()
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
if __name__ == '__main__':
    def mainwork():
        global_parameters = {}
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()
        for i in range(num_comm):
            global_parameters= env_geneWk(global_parameters,i)

    data_server = GetDataSet()
    data_server.load_data()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = LinearNet()
    net = net.to(dev)
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function = loss_function.to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    myClients = ClientsGroup(num_of_client, dev)
    # valDataLoader = myClients.val_data_loader
    test_label = torch.FloatTensor(data_server.val_label)
    test_data = torch.FloatTensor(data_server.val_data)
    testDataLoader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
    mainwork()