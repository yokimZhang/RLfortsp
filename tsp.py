import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import os


# define Reward
def reward(sample_solution, USE_CUDA=True):
    """
    :param sample_solution: List of length sourceL of [batch_size] Tensors, [seq_len x batch_size x input_size]
    :return: Tensor of shape [batch_size] contains rewards
    """

    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)    # denotes n cities
    tour_len = torch.zeros([batch_size])

    if USE_CUDA:
        tour_len = tour_len.cuda()

    for i in range(n-1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i+1], dim=1)

    tour_len += torch.norm(sample_solution[n-1] - sample_solution[0], dim=1)      # 计算路径长度

    return tour_len


# read dataset of paper
def read_paper_dataset(path):
    x, y = [], []
    with open(path) as fp:
        for line in tqdm(fp):
            inputs, outputs = line.split('output')
            x.append(np.array(inputs.split(), dtype=np.float32)).reshape(-1, 2)
            y.append(np.array(outputs.split(), dtype=np.int32)[:-1])   # skip the last one

    return x, y


# randomly generate data
class TSPDataset(Dataset):
    """
    data_set: [batch_size x seq_len x input_dim]
    """

    def __init__(self, filename=None, seq_len=10, num_samples=1000000, random_seed=111):
        super(TSPDataset, self).__init__()
        torch.manual_seed(random_seed)
        # [num_samples x seq_len x input_dim]
        self.data_set = []
        if filename:
            with open(filename) as fp:
                for line in tqdm(fp):
                    x = np.array(line.split('output')[0].split(), dtype=np.float32).reshape(-1, 2)
                    self.data_set.append(x)
        else:
            # randomly sample points uniformly from [0, 1]
            for _ in tqdm(range(num_samples)):
                x = torch.FloatTensor(seq_len, 2).uniform_(0, 1)
                self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

