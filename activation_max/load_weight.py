import torch
from torch.autograd import Variable

from activation_max.model import MNIST_DNN

model = MNIST_DNN()
model.load_state_dict(torch.load('MNIST_CNN.pkl'))
