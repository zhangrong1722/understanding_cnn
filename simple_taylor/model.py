import torch
from torch.autograd import Variable
import torch.nn as nn

class MNIST_DNN(nn.Module):
    def __init__(self):
        super(MNIST_DNN, self).__init__()
        self.dense1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU(True)

        self.dense2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU(True)

        self.dense3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU(True)

        self.dense4 = nn.Linear(512, 512)
        self.relu4 = nn.ReLU(True)

        self.dense5 = nn.Linear(512, 10)

        # set the bias of Linea layer to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)


    def forward(self, x):
        x = self.dense1(x)
        x = self.relu1(x)

        x = self.dense2(x)
        x = self.relu2(x)

        x = self.dense3(x)
        x = self.relu3(x)

        x = self.dense3(x)
        x = self.relu3(x)

        x = self.dense4(x)
        x = self.relu4(x)

        x = self.dense5(x)
        return x

if __name__ == '__main__':
    model = MNIST_DNN()
