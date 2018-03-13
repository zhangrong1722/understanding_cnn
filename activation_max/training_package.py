import torch
import argparse
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from activation_max.model import MNIST_DNN
from utils.utils import exp_lr_scheduler

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--batch_size', default=32, type=int,
                    help='mini-batch size(default:32)')
parser.add_argument('--num_works', default=4, type=int,
                    help='threading nums when reading dataset(default:4)')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run(default:50)')
parser.add_argument('--init_lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--cuda', default=True if torch.cuda.is_available() else False, type=bool,
                    help='save variable in GPU')


def save():
    pass
    # torch.save(model.state_dict(), 'MNIST_CNN.pkl')


def main():
    global args
    args = parser.parse_args()
    train_mnist = datasets.MNIST('../dataset/mnist', train=True, download=True, transform=transforms.ToTensor())
    test_mnist = datasets.MNIST('../dataset/mnist', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_mnist, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_mnist, batch_size=args.batch_size, shuffle=False)

    model = MNIST_DNN()
    if args.cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum)
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        evl(model, test_loader, criterion)
    save()


def train(loader, model, criterion, optimizer, epoch):
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    print('epoch {0}'.format(epoch + 1))
    print('*' * 10)
    for idx, (inputs, targets) in enumerate(loader, 1):
        inputs = inputs.view(inputs.size(0), -1)
        if args.cuda:
            inputs = Variable(inputs).cuda()
            targets = Variable(targets).cuda()
        else:
            inputs = Variable(inputs)
            targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.data[0] * targets.size(0)
        _, pred = torch.max(outputs, 1)
        num_correct = (pred == targets).sum()
        running_acc += num_correct.data[0]
        total += targets.size(0)
        optimizer = exp_lr_scheduler(optimizer=optimizer, epoch=epoch + 1, init_lr=args.init_lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 500 == 0:
            print('{}/{} Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, args.epochs, running_loss / total,
                                                           running_acc / total))

    print('Finish {} epochs, Loss:{:.6f}, Acc:{:.6f}'.format(epoch + 1,
                                                             running_loss / total,
                                                             running_acc / total))


def evl(model, loader, criterion):
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    total = 0
    for (inputs, targets) in loader:

        inputs = inputs.view(inputs.size(0), -1)
        if args.cuda:
            inputs = Variable(inputs, volatile=True).cuda()
            targets = Variable(targets, volatile=True).cuda()
        else:
            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        loss = criterion(outputs, targets)
        eval_loss += loss.data[0]

        num_correct = (pred == targets).sum()
        eval_acc += num_correct.data[0]
        total += targets.size(0)

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / total, eval_acc / total))


if __name__ == '__main__':
    main()
