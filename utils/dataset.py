from torchvision import datasets,transforms
from torch.utils.data import DataLoader

train_mnist = datasets.MNIST('../dataset/mnist', train=True, download=True, transform=transforms.ToTensor())
test_mnist = datasets.MNIST('../dataset/mnist', train = False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_mnist, batch_size=1, shuffle=True)
test_loader = DataLoader(test_mnist, batch_size=1, shuffle=False)

assert len(train_loader) == 60000
assert len(test_loader) == 10000
assert len(test_mnist) == 10000
assert len(train_mnist) == 60000


# assert len(train_loader) == 50000

for i,data in enumerate(train_loader, 1):
    img,label = data
    img = img.view(img.size(0), -1)
    assert img.size(1) == 784
    if i == 5:
        break
