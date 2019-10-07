import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
dataset=torch.randint(1,10,[2,2,2]).float().uniform_(0,1)
print(dataset)