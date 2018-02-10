from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch

train_mnist = datasets.MNIST('../dataset/mnist', train=True, download=True, transform=transforms.ToTensor())
test_mnist = datasets.MNIST('../dataset/mnist', train = False, download=True, transform=transforms.ToTensor())

batch_size = 32
train_loader = DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_mnist, batch_size=batch_size, shuffle=False)

assert len(test_mnist) == 10000
assert len(train_mnist) == 60000
# size of image:(channels, height, width)
assert (1, 28, 28) == (train_mnist[0][0]).size()
# the type of image
assert type(train_mnist[0][0]) == torch.FloatTensor

for i,data in enumerate(train_loader, 1):
    # <class 'torch.LongTensor'>
    img,label = data
    assert img.size() == (batch_size, 1, 28, 28)
    img = img.view(img.size(0), -1)
    assert img.size(1) == 784
    if i == 1:
        break
