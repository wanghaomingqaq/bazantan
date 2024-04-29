"""
只同步一次参数
"""
import sys
from FashionMNIST import FashionMNIST
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn

epoch = 3
batchsize = 8
learning_rate = 0.0001
num_comm = 20000
num_of_client = 10
num_in_comm = 10  # 每次抽取个数


def server_train(global_parameters):
    net.load_state_dict(global_parameters, strict=True)
    for data, label in waterDataLoader:
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        loss = loss_function(preds, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return net.state_dict()

def add_gaussian_noise_to_gradients(model, mu=0.5, sigma=2e-6):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.normal(mean=mu, std=sigma, size=param.grad.shape, device=param.grad.device)
            param.grad += noise


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 13)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        #
        # # 应用 Softmax 函数以获取预测概率
        # probabilities = F.softmax(tensor, dim=1)
        # # 判断是否达到阈值
        # max_probs, predictions = torch.max(probabilities, dim=1)
        # threshold = 0.1  # 设置阈值，这个阈值可以根据实际情况调整
        # predictions[max_probs < threshold] = 12  # 假设第11个类别是 NULL
        return tensor


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

    def localUpdate(self, localSteps, localBatchSize, Net, loss_function, opti, global_parameters, attack=False):
        Net.load_state_dict(global_parameters, strict=True)

        # 重新打乱并加载本地数据
        self.train_ds = TensorDataset(torch.tensor(self.train_x), torch.tensor(self.train_y))
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        step = 0
        for data, label in self.train_dl:
            step += 1
            data, label = data.to(self.dev), label.to(self.dev)
            preds = Net(data)
            loss = loss_function(preds, label)
            loss.backward()
            opti.step()
            opti.zero_grad()
            if step >= localSteps:
                break
        if attack:
            add_gaussian_noise_to_gradients(Net)
            state_dict = Net.state_dict()
            for key in state_dict:
                state_dict[key] = state_dict[key]
            return state_dict
        return Net.state_dict()


class ClientsGroup(object):
    def __init__(self, numOfClients, dev, data_dis='imbalance'):
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
        data = FashionMNIST(100, group_labels)
        train_datasets = data.get_train_dataset()
        for i in range(num_of_client):
            someone = client(train_datasets[i][0], train_datasets[i][1], self.dev)
            #
            # print(someone.train_x.shape)
            self.clients_set['client{}'.format(i)] = someone


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


def env_geneWk(a_t, global_parameters, comn):
    # start_time = time.time()
    clients_in_comm = ['client{}'.format(i) for i in range(num_in_comm)]
    sum_parameters = None
    attack = False
    for idx, client in enumerate(clients_in_comm):
        # 本地更新
        if idx >= 9:
            attack = True
        local_parameters = myClients.clients_set[client].localUpdate(
            epoch, batchsize, net, loss_function, optimizer, global_parameters, attack=attack)

        # 使用动作 a_t 权重更新参数
        weight = a_t[idx]
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


if __name__ == '__main__':
    def mainwork():
        global_parameters = {}
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()
        global_parameters = server_train(global_parameters)
        for i in range(num_comm):
            action = [0.1] * num_of_client
            print("action:", action)
            global_parameters = env_geneWk(action, global_parameters, i)
            global_parameters = server_train(global_parameters)


    data_server = FashionMNIST(100, group_labels=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

    testdata, testLabel = data_server.get_test_dataset()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = CNN()
    net = net.to(dev)
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function = loss_function.to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    myClients = ClientsGroup(num_of_client, dev)
    testDataLoader = DataLoader(TensorDataset(testdata, testLabel), batch_size=100, shuffle=False)

    waterData, waterLabel = data_server.get_water_dataset()
    waterDataLoader = DataLoader(TensorDataset(waterData, waterLabel), batch_size=2, shuffle=True);

    mainwork()
