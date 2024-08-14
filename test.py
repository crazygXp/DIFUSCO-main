#
# from pyconcorde.concorde.tsp import TSPSolver
# from typing import *
# from random import random
# import matplotlib.pyplot as plt
# import numpy as np
#
# def tsp(dots:List[Tuple[float, float]])->List[int]:
#     """
#     TSP最优算法（concorde）
#     :param dots: 一系列的点的坐标，点之间的距离表示代价
#     :return: 一系列点的编号，代表得到的哈密顿环
#     """
#     # solver only supports integral edge weights, so here float will be rounded to two
#     # decimals, multiplied by 100 and converted to integer
#     xs = []
#     ys = []
#     for (x, y) in dots:
#         xs.append(int(x * 1000))
#         ys.append(int(y * 1000))
#     solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
#     solution = solver.solve()
#     #print(solution.found_tour)
#     #print(solution.optimal_value)
#     #print(solution.tour)
#     return solution.tour.tolist()
#
# class ConcordeTsp(object):
#     """
#     TSP算法封装类
#     使用方法：ConcordeTsp(dots).tsp()
#     其中，dots为二维平面上的点的列表
#     """
#     def __init__(self, dots:List[Tuple[float, float]]):
#         # 去除重复的点
#         dotSet = set()
#         dotFilted = []
#         for dot in dots:
#             if not dot in dotSet:
#                 dotSet.add(dot)
#                 dotFilted.append(dot)
#         self.__dots = dotFilted
#
#     def tsp(self)->List[int]:
#         """
#         :return: tsp回路的近似解，返回结果不包含初始结点
#         """
#         return tsp(self.__dots)
#
# if __name__ == "__main__":
#     """以下是测试代码"""
#     dots = []
#     xs = []
#     ys = []
#     for i in range(51):
#         x = random()
#         y = random()
#         dots.append((x, y))
#         xs.append(x)
#         ys.append(y)
#     path = ConcordeTsp(dots).tsp()
#     print(path)
#     path.append(path[0])
#     plt.scatter(np.array(xs)[path], np.array(ys)[path])
#     for i in range(len(xs)):
#         plt.annotate(str(path[i]),
#                      xy=(xs[path[i]], ys[path[i]]),
#                      xytext=(xs[path[i]] + 0.0001, ys[path[i]] + 0.0001))
#         # 这里xy是需要标记的坐标，xytext是标签显示的位置
#     plt.plot(np.array(xs)[path], np.array(ys)[path])
#     plt.show()

# import torch
# print(torch.cuda.is_available())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import wandb


batch_size = 200
learning_rate = 0.01
epochs = 30

wandb.init(
    # Set the project where this run will be logged
    project=" ",#写自己的
    entity=" ",#写自己的
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "MLP",
        "dataset": "MNIST",
        "epochs": epochs,
    })

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x


device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)


global_step = 0

for epoch in range(epochs):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        global_step += 1

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            #展示数据
            wandb.log({"Train loss": loss.item()})
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    wandb.log({"Test avg loss":test_loss,"Test acc": 100. * correct / len(test_loader.dataset)})

wandb.finish()
