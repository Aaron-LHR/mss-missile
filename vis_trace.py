import os
import numpy as np
import matplotlib
import torch
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
import torch
import copy
from tqdm import tqdm, trange
import math

# from torch.utils.tensorboard import SummaryWriter
work_dir = "./"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device


class Curve_Model_All_In_One(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(7, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 4096)
        self.linear4 = nn.Linear(4096, 3)


    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear3(x)
        x = torch.nn.functional.relu(x)
        x = self.linear4(x)

        return x
    
    
    
data_dir = f'{work_dir}data/data07081/'

# 时间, 起点b、l、h, 终点b、l、h 
# 发射系坐标
data_files = os.listdir(data_dir)[:1]
x_index=7
batch_size = 26240
model = torch.load("model/Curve_Model_All_In_One-learn_rate_1e-06-mse_loss-adam-epoch_97-loss_346846928960.0-last_lr_ 0.0000010000-err_ 840443.7089612312.chpk")
model.to(device)
model.eval()
with torch.no_grad():
    for i in trange(len(data_files)):
        file = data_files[i]
        single_trace = np.loadtxt(os.path.join(data_dir, file), dtype=np.float32)[:-2]
        start_point = single_trace[0, x_index: x_index + 3]
        end_point = single_trace[-1, x_index: x_index + 3]
        feature_list = []
        for j, one_time_data in enumerate(single_trace):
            time = one_time_data[:1]
            feature_list.append(np.concatenate((time, start_point, end_point)))
        pred_trace = []
        for feature in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(np.stack(feature_list))), batch_size=batch_size, sampler=None):
            pred_points = model(feature).to('cpu')
            pred_trace.append(pred_points)
        pred_trace = np.stack(pred_trace)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(xs=single_trace[:, x_index], ys=single_trace[:, x_index + 1], zs=single_trace[:, x_index + 2])
        fig.show()

        # 轨迹
        ax.plot(xs=pred_trace[:, 0], ys=pred_trace[:, 1], zs=pred_trace[:, 2])
        # 起点
        ax.scatter(start_point[0], start_point[1], start_point[2], 'r')
        fig.show()