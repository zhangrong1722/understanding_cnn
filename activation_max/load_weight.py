import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np

from activation_max.model import MNIST_DNN

model = MNIST_DNN()
weights = torch.load('MNIST_CNN.pkl')
model.load_state_dict(weights)


def make_prediction():
    img = Variable(torch.from_numpy(np.array(Image.open('../dataset/test_imgs/3.png'))))
    img = img.float()
    img = img.view(1, 784)

    pred = model(img)
    pred = pred.data.numpy()
    print('the final result is ',np.argmax(pred))

def parameters(model):
    for para in list(model.parameters()):
        para.requires_grad = False

# how to load part of pre-trained model
def show_weight(weights):
    for name, para in weights.items():
        print(name)

if __name__ == '__main__':
    parameters(model)