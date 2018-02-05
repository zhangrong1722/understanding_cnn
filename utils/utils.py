import torch
from torchvision import datasets,transforms
import numpy as np

def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def set_parameters_static(model):
    for para in list(model.parameters()):
        para.requires_grad = False
    return model


def get_img_means():
    train_mnist = datasets.MNIST('../dataset/mnist', train=True, download=True, transform=transforms.ToTensor())

    mnist_imgs = [data[0].view(1, 784).numpy() for data in train_mnist]
    mnist_labels = [data[1] for data in train_mnist]

    imgs_means = np.zeros((10, 784))
    for i in range(10):
        imgs = []
        for index, label in enumerate(mnist_labels):
            if label == i:
                imgs.append(mnist_imgs[index])
        imgs_means[i, :] = np.mean(imgs, axis=0)
    return torch.from_numpy(imgs_means).type(torch.FloatTensor)
