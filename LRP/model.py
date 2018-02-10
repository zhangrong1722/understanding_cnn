import torch
from torch.autograd import Variable

import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class MNIST_DNN(nn.Module):
    def __init__(self):
        super(MNIST_DNN, self).__init__()
        self.dense1 = nn.Linear(784, 300)
        self.relu1 = nn.ReLU(True)

        self.dense3 = nn.Linear(300, 10)

    def forward(self, x):
        dense1 = self.dense1(x)
        dense1 = self.relu1(dense1)

        logits = self.dense3(dense1)
        predict = F.softmax(logits, dim=1)

        return [x, dense1, predict], logits


class LRP(object):
    def __init__(self, alpha, activations, weights, bias):
        self.alpha = alpha
        if len(activations) != 3:
            raise RuntimeError('this code is only applied in the model architecture defined in file model.py.')
        activations = [activation.data.numpy()
                       for activation in activations if type(activation) is not np.ndarray]
        self.activations = activations
        self.activations.reverse()

        self.weights = weights
        self.weights.reverse()

        self.bias = bias
        self.bias.reverse()

    def __call__(self, logit):
        Rs = []
        # Rs.append(self.activations[0][:,logit,None])
        # assert self.activations[1].shape == (1, 300)
        # assert self.weights[0][logit, :, None].shape == (300, 1)
        #
        # assert self.bias[0][logit, None].shape == (1,)
        # assert Rs[-1].shape == (1,1)
        #
        # Rs.append(self.backprop_dense(self.activations[1],self.weights[0][logit, :, None],
        #                               self.bias[0][logit,None],Rs[-1]))
        #
        # assert self.activations[2].shape == (1,784)
        # assert np.transpose(self.weights[1]).shape == (784,300)
        # assert self.bias[1].shape == (300,)
        # assert Rs[-1].shape == (1,300)
        #
        # Rs.append(self.backprop_dense(self.activations[2], np.transpose(self.weights[1]),
        #                               self.bias[1], Rs[-1]))

        # express in for for cycle
        j = 0
        for i in range(len(self.activations) - 1):
            if i == 0:
                Rs.append(self.activations[i][:, logit, None])

                Rs.append(self.backprop_dense(
                    self.activations[i + 1], self.weights[j][logit, :, None], self.bias[j][logit, None], Rs[-1]))
            else:
                Rs.append(
                    self.backprop_dense(self.activations[i + 1], np.transpose(self.weights[j]), self.bias[j], Rs[-1]))
            j += 1
        return Rs[-1]

    # note:the type of all parameters are numpy.ndarray(using numpy)
    def backprop_dense(self, activation, weighs, bias, relevance):
        w_p = np.maximum(weighs, 0)
        b_p = np.maximum(bias, 0)

        z_p = np.matmul(activation, w_p) + b_p
        s_p = relevance / z_p
        c_p = np.matmul(s_p, np.transpose(w_p))

        w_n = np.minimum(weighs, 0)
        b_n = np.minimum(bias, 0)

        z_n = np.matmul(activation, w_n) + b_n
        s_n = relevance / z_n
        c_n = np.matmul(s_n, np.transpose(w_n))

        return activation * (self.alpha * c_p + (1 - self.alpha) * c_n)
