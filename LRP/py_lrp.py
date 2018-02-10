import torch
from torch.autograd import Variable
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from LRP.model import MNIST_DNN,LRP
from utils.utils import get_all_digit_imgs

def get_model_parameters(model):
    weights = []
    bias = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.data.numpy())
            bias.append(layer.bias.data.numpy())
    return weights,bias

num = 7
model = MNIST_DNN()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('weights.pkl'))
weights,bias = get_model_parameters(model)

sample_imgs = get_all_digit_imgs(is_feed_cnn = False)

if torch.cuda.is_available():
    sample_imgs = Variable(sample_imgs).cuda()
else:
    sample_imgs = Variable(sample_imgs)

# activation: the output pass ReLU of every layer
# input.size=(1,784)
activations, _ = model(sample_imgs[num][None,:])

lrp = LRP(alpha=1, activations=activations, weights=weights, bias=bias)

Rs = lrp(num)

plt.figure(figsize=(4, 4))
plt.subplot()
plt.imshow(np.reshape(Rs, [28, 28]), cmap='hot_r')
plt.title('Digit: {}'.format(num))
plt.colorbar()
plt.show()
