from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import numpy as np
import torch

train_mnist = datasets.MNIST('dataset/mnist', train=True, download=True, transform=transforms.ToTensor())
# test_mnist = datasets.MNIST('dataset/mnist', train=False, download=True, transform=transforms.ToTensor())

# train_loader = DataLoader(train_mnist, batch_size=len(train_mnist), shuffle=True)
# test_loader = DataLoader(test_mnist, batch_size=10, shuffle=False)

mnist_imgs = [data[0].view(1, 784).numpy() for data in train_mnist]
mnist_labels = [data[1] for data in train_mnist]

imgs_means = np.zeros((10, 784))
# nums = []
for i in range(10):
    imgs = []
    for index, label in enumerate(mnist_labels):
        if label == i:
            imgs.append(mnist_imgs[index])
    # nums.append(len(imgs))
    imgs_means[i, :]=np.mean(imgs, axis=0)

# assert 60000 == sum(nums)

# imgs_means = torch.from_numpy([mean.reshape for mean in imgs_means])
# assert type(imgs_means) == torch.FloatTensor
print()
