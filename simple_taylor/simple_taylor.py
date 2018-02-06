import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np

from simple_taylor.model import MNIST_DNN
from utils.utils import get_all_digit_imgs,pixel_range


model = MNIST_DNN()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('weights.pkl'))

imgs = get_all_digit_imgs(is_feed_cnn = False)
if torch.cuda.is_available():
    imgs = Variable(imgs, requires_grad=True).cuda()
else:
    imgs = Variable(imgs, requires_grad=True)

logits = model(imgs)

logits.backward(torch.ones(logits.size()))

gradients = imgs.grad.data.numpy()
sample_imgs = imgs.data.numpy()
assert gradients.shape == (10, 784)
# element-wise product
hmaps = np.negative(sample_imgs * gradients)
assert hmaps.shape == (10, 784)

# display visulization result
plt.figure(figsize=(7, 7))
for i in range(5):
    plt.subplot(5, 2, 2 * i +1)
    plt.imshow(np.reshape(sample_imgs[2 * i,:], (28, 28)), cmap='gray')

    vmin, vmax = pixel_range(hmaps[2 * i, :])
    plt.imshow(np.reshape(gradients[2 * i,:], (28, 28)), vmin=vmin, vmax=vmax, cmap='bwr', alpha=0.5)
    plt.title('Digit:{}'.format(2 * i))

    plt.subplot(5, 2, 2 * i + 2)
    plt.imshow(np.reshape(sample_imgs[2 *i + 1, :], (28, 28)), cmap='gray')

    vmin, vmax = pixel_range(hmaps[2 * i+1, :])
    plt.imshow(np.reshape(gradients[2*i + 1, :], (28, 28)), vmin=vmin, vmax=vmax, cmap='bwr', alpha=0.5)
    plt.title('Digit:{}'.format(2 * i + 1))

plt.tight_layout()
plt.show()


