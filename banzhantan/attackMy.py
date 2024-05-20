"""
只同步一次参数
"""
import copy
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


def server_train(parameters,epoch):
    net.load_state_dict(parameters, strict=True)
    for i in range(epoch):
        for data, label in waterDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            loss = loss_function(preds, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return net.state_dict()

def server_test(local_parameter):
    sum_accu = 0
    num = 0
    net.load_state_dict(local_parameter, strict=True)
    for data, label in waterDataLoader:
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        # print(preds)
        preds = torch.argmax(preds, dim=1)
        # print(preds)
        sum_accu += (preds == label).float().mean()
        num += 1
    return sum_accu / num
# def add_gaussian_noise_to_gradients(model, mu=0.5, sigma=2e-6):
#     with torch.no_grad():
#         for param in model.parameters():
#             noise = torch.normal(mean=mu, std=sigma, size=param.grad.shape, device=param.grad.device)
#             param.grad += noise
def add_gaussian_noise_to_gradients(model, mu=0.002, sigma=2e-6):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.size(), device=param.device) * sigma + mu  # 使用randn生成标准正态分布，然后调整均值和方差
            param += noise
    return model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
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
            model = add_gaussian_noise_to_gradients(Net)
            return model.state_dict()
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
        group_labels = [[0, 9], [0, 9], [0, 9], [0, 9], [0, 9]]  # Define label groups
        data = FashionMNIST(num_of_client, group_labels)
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


def normalize_non_zero(lst):
    # 计算非零元素的数量
    non_zero_count = lst.count(False)
    if non_zero_count == 0:
        return [0.1]*10  # 如果没有非零元素，直接返回原列表

    # 每个非零元素的新值
    new_value = 1 / non_zero_count

    # 更新列表，非零值设置为new_value
    return [new_value if x != True else 0 for x in lst]


def env_geneWk(a_t, global_parameters, comn,prev_params, global_mask, attackCome=False):
    # 初始化
    clients_in_comm = ['client{}'.format(i) for i in range(num_in_comm)]
    sum_parameters = None
    attack = False
    total_acc = []
    total_parameters = []

    # 收集所有客户端的本地参数和精度
    isAttack = False
    for idx, client in enumerate(clients_in_comm):
        if attackCome:
            attack = idx <= 0
        else:
            attack = False
        init_para = copy.deepcopy(global_parameters)
        # init_para = global_parameters

        local_parameters = myClients.clients_set[client].localUpdate(
            epoch, batchsize, net, loss_function, optimizer, init_para, attack=attack)
        if attackCome:
            isAttack = compareParam(copy.deepcopy(local_parameters),global_mask,prev_params)
        # acc_ = server_test(local_parameters)
        total_acc.append(isAttack)
        total_parameters.append(copy.deepcopy(local_parameters))

    # 根据性能结果计算新的权重
    new_weight = normalize_non_zero(total_acc)
    print("acc:", total_acc)
    new_weight = [0.1]*10
    print("action: ", new_weight)
    # 加权平均参数更新
    for idx, local_parameters in enumerate(total_parameters):
        weighted_parameters = {key: new_weight[idx] * var.clone() for key, var in local_parameters.items()}
        if sum_parameters is None:
            sum_parameters = weighted_parameters
        else:
            for var in sum_parameters:
                sum_parameters[var] += weighted_parameters[var]

    # 标准化全局参数
    for var in global_parameters:
        global_parameters[var] = sum_parameters[var]

    # 执行额外的动作或处理
    acc(comn)

    return global_parameters

def flatten_params(model):
    return torch.cat([p.flatten() for p in model.parameters() if p.requires_grad])

def jaccard_similarity(tensor1, tensor2):
    set1, set2 = set(tensor1.numpy()), set(tensor2.numpy())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def compareParam(clientParam, global_mask,prev_params):
    unchanged_mask = torch.ones_like(prev_params, dtype=torch.bool)
    sensitivity_threshold = 0.00001  # 设定阈值
    with torch.no_grad():
        tmpNet = CNN()
        tmpNet.load_state_dict(clientParam)
        tmp = flatten_params(tmpNet)
        changes = torch.abs(tmp - prev_params) < sensitivity_threshold
        unchanged_mask &= changes
    long_term_unchanged_elements = torch.where(unchanged_mask)[0]

    # print("client 长期没有变化的参数元素的索引:", long_term_unchanged_elements)
    ret = jaccard_similarity(long_term_unchanged_elements,global_mask)
    print(ret,"相似度")
    if ret > 0.1:
        return False
    else:
        return True
if __name__ == '__main__':
    def mainwork():
        global_parameters = {}
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()
        prev_params = flatten_params(net)
        sensitivity_threshold = 0.00001  # 设定阈值
        unchanged_mask = torch.ones_like(prev_params, dtype=torch.bool)
        long_term_unchanged_elements = torch.where(unchanged_mask)[0]
        for i in range(num_comm):
            action = [0.1] * num_of_client
            attackCom = False
            if i> 50:
                attackCom = True
            global_parameters = env_geneWk(action, global_parameters, i,prev_params,long_term_unchanged_elements, attackCome=attackCom)
            net.load_state_dict(global_parameters)
            if i< 49:
                with torch.no_grad():
                    current_params = flatten_params(net)
                    changes = torch.abs(current_params - prev_params) < sensitivity_threshold
                    unchanged_mask &= changes
                # 显示哪些参数元素长期没有变化
                long_term_unchanged_elements = torch.where(unchanged_mask)[0]
                # print("长期没有变化的参数元素的索引:", long_term_unchanged_elements)
                # print(len(long_term_unchanged_elements))

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
