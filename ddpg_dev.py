"""
只同步一次参数
"""
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
g_exporStep = 1  # 第一步随机探索
g_num_steps = 4,  # 进行学习的频次
g_memory_size = 500,  # 经验回放池的容量
g_replay_start_size = 100,  # 开始回放的次数
g_update_target_steps = 100,  # 同步参数的次数
g_gamma = 0.9,
epoch = 2
batchsize = 8
learning_rate = 0.0001
num_comm = 10000
num_of_one_epoch = 200
num_of_client = 108
num_in_comm = 108  # 每次抽取个数
# g_train_data = r'D:\github\workspace\pytorch_study\dataset\108resnetv2_train.csv'
# g_test_data = r'D:\github\workspace\pytorch_study\dataset\108resnetv2_test.csv'
# g_val_data = r'D:\github\workspace\pytorch_study\dataset\108resnetv2_val.csv'
g_train_data = './108resnetv2_train.csv'
g_test_data = './108resnetv2_test.csv'
g_val_data = './108resnetv2_val.csv'

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

    def dataSetBalanceAllocation(self):
        if self.data_dis == 'balance':
            mnistDataSet = GetDataSet()
            mnistDataSet.load_data()
            val_label = torch.FloatTensor(mnistDataSet.val_label)
            val_data = torch.FloatTensor(mnistDataSet.val_data)
            self.val_data_loader = DataLoader(TensorDataset(val_data, val_label), batch_size=100, shuffle=False)
            train_data = mnistDataSet.train_data
            train_label = mnistDataSet.train_label
            shard_size = mnistDataSet.train_data_size // num_of_client
            for i in range(num_of_client):
                shards_id1 = i * shard_size
                shards_id2 = (i + 1) * shard_size
                local_data = train_data[shards_id1: shards_id2]
                local_label = train_label[shards_id1: shards_id2]
                someone = client(local_data, local_label, self.dev)
                self.clients_set['client{}'.format(i)] = someone
        else:
            self.get_imbalance()
    def get_imbalance(self):
        mnistDataSet = GetDataSet()
        mnistDataSet.load_data()
        val_label = torch.FloatTensor(mnistDataSet.val_label)
        val_data = torch.FloatTensor(mnistDataSet.val_data)
        self.val_data_loader = DataLoader(TensorDataset(val_data, val_label), batch_size=100, shuffle=False)
        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label
        shard_size = mnistDataSet.train_data_size // num_of_client
        for i in range(int(num_of_client/2)):
            shards_id1 = i * shard_size
            shards_id2 = (i + 1) * shard_size
            local_data = train_data[shards_id1: shards_id2-4]
            local_label = train_label[shards_id1: shards_id2-4]
            someone = client(local_data, local_label, self.dev)
            self.clients_set['client{}'.format(i)] = someone
        for i in range(int(num_of_client/2),num_of_client):
            shards_id1 = i * shard_size
            shards_id2 = (i + 1) * shard_size
            local_data = train_data[shards_id2-4: shards_id2]
            local_label = train_label[shards_id2-4: shards_id2]
            someone = client(local_data, local_label, self.dev)
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
    for data, label in valDataLoader:
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        preds = torch.argmax(preds, dim=1)
        sum_accu += (preds == label).float().mean()
        num += 1
    print("\n" + 'accuracy: {}'.format(sum_accu / num))
    print("\n" + 'comm: {}'.format(com))

def env_geneWk(a_t,global_parameters,comn):
    # start_time = time.time()
    clients_in_comm = ['client{}'.format(i) for i in range(num_in_comm)]
    sum_parameters = None
    # 对每个客户端执行动作
    l_b = []
    l_a = []
    n_k = []
    diff = []
    n_clients = len(clients_in_comm)  # 获取客户端数量
    sum_parameters = None

    for idx, client in enumerate(clients_in_comm):
        # 本地更新
        local_parameters, loss_b, loss_a, k_n = myClients.clients_set[client].localUpdate(
            epoch, batchsize, net, loss_function, optimizer, global_parameters)

        # 使用动作 a_t 权重更新参数
        weight = a_t[idx]
        weighted_parameters = {key: weight * var.clone() for key, var in local_parameters.items()}

        if sum_parameters is None:
            sum_parameters = weighted_parameters
        else:
            for var in sum_parameters:
                sum_parameters[var] += weighted_parameters[var]
        # print(loss_b)
        l_a.append(loss_a)
        l_b.append(loss_b)
        n_k.append(k_n)
        diff.append(loss_a-loss_b)
    # 将累加的参数除以客户端数量以取平均
    acc(comn)
    for var in sum_parameters:
        sum_parameters[var] = sum_parameters[var]


    # for idx, client in enumerate(clients_in_comm):
    #     # 本地更新
    #     local_parameters, loss_b, loss_a, k_n = myClients.clients_set[client].localUpdate(
    #         epoch, batchsize, net, loss_function, optimizer, global_parameters)
    #     # 使用动作 a_t 权重更新参数
    #     weight = a_t[idx]
    #     if sum_parameters is None:
    #         sum_parameters = {key: var.clone() for key, var in local_parameters.items()}
    #     else:
    #         for var in sum_parameters:
    #             sum_parameters[var] = (1 - weight) * sum_parameters[var] + weight * local_parameters[var]
    # 更新全局参数
    for var in global_parameters:
        global_parameters[var] = sum_parameters[var]
    r_t = get_reward(l_a, l_b)
    s_next = l_b + l_a + n_k + diff
    s_next = np.array(s_next)
    return s_next,-r_t,global_parameters
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
        state_dim = 4*num_of_client
        hidden_dim = 2*num_of_client
        action_dim = num_of_client
        ddpg = DDPG(action_dim, state_dim, hidden_dim)
        # action_space = spaces.Box(low=np.array([0.0] * action_dim), high=np.array([1.0] * action_dim), dtype=np.float32)
        # ou_noise = OUNoise(action_space,decay_period=1000)
        batch_size = 8
        state = np.arange(state_dim)
        for i in range(num_comm):
            epsilon = min(0.99, 0.01 + 0.99 * (i/ 2000))
            if np.random.rand() < epsilon:
                action = ddpg.policy_net.get_action(state)
            else:
                random_values = np.random.rand(108)
                action = softmax(random_values)
            # action = ou_noise.get_action(action,i)
            # 归一化
            # action = action / action.sum()
            print("action:",action)
            next_state,reward,global_parameters= env_geneWk(action,global_parameters,i)
            print(reward)
            ddpg.replay_buffer.push(state,action,reward,next_state)
            if len(ddpg.replay_buffer) > batch_size:
                ddpg.ddpg_update()
            state = next_state

    data_server = GetDataSet()
    data_server.load_data()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = LinearNet()
    net = net.to(dev)
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function = loss_function.to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    myClients = ClientsGroup(num_of_client, dev)
    valDataLoader = myClients.val_data_loader
    test_label = torch.FloatTensor(data_server.test_label)
    test_data = torch.FloatTensor(data_server.test_data)
    testDataLoader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
    mainwork()