import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def test_numpy():
    global img_means
    labels = np.array([1, 2, 2, 3, 3, 2])
    images = np.linspace(0, 59, 60).reshape((6, 10))
    print(images)
    img_means = []
    img_means.append(np.mean(images[labels == 2], axis=0))
    print(img_means)


def get_img_means():
    global img_means
    train_mnist = datasets.MNIST('../dataset/mnist', train=True, download=True, transform=transforms.ToTensor())
    mnist_imgs = np.array([data[0].view(784, ).numpy() for data in train_mnist])
    mnist_labels = np.array([data[1] for data in train_mnist])
    assert (mnist_labels).shape == (60000,)
    assert (mnist_imgs).shape == (60000, 784)
    img_means = []
    for i in range(10):
        img_means.append(np.mean(mnist_imgs[mnist_labels == i], axis=0))
    return np.array(img_means)


def cacul_img_means():
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
    return np.array(imgs_means)


if __name__ == '__main__':
    # img_means =get_img_means()
    img_means=cacul_img_means()
    plt.imshow(img_means[1,:].reshape((28,28)), cmap='gray')
    plt.show()