import os

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tensorflow.python.ops import nn_ops, gen_nn_ops
import tensorflow as tf
import numpy as np


class MNIST_DNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            dense1 = tf.layers.dense(inputs=X, units=300, activation=tf.nn.relu, use_bias=True)
            logits = tf.layers.dense(inputs=dense1, units=10, activation=None, use_bias=True)
            prediction = tf.nn.softmax(logits)

        return [dense1, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class LRP:

    def __init__(self, alpha, activations, weights, biases, name):
        self.alpha = alpha
        self.activations = activations
        self.activations.reverse()

        self.weights = weights
        self.weights.reverse()

        self.biases = biases
        self.biases.reverse()

        self.name = name

    def __call__(self, logit):

        with tf.name_scope(self.name):
            Rs = []
            j = 0
            for i in range(len(self.activations) - 1):
                if i is 0:
                    Rs.append(self.activations[i][:, logit, None])
                    Rs.append(self.backprop_dense(self.activations[i + 1], self.weights[j][:, logit, None],
                                                  self.biases[j][logit, None], Rs[-1]))
                    j += 1
                    continue
                elif 'dense' in self.activations[i].name.lower():
                    # self.activatios[i+1]:(None,784)
                    # self.weights[j]:(784,300)
                    # self.biases[j]:(300)
                    # Rs[-1]:(1,300)
                    Rs.append(self.backprop_dense(self.activations[i + 1], self.weights[j], self.biases[j], Rs[-1]))
                    j += 1
                    print('index=',i)

            return Rs[-1]

    def backprop_dense(self, activation, kernel, bias, relevance):
       # activation: (1,300) relevance:(1,1)
       # kernel:(300,1) bias:(1,1)
        W_p = tf.maximum(0., kernel)
        b_p = tf.maximum(0., bias)
        z_p = tf.matmul(activation, W_p) + b_p
        s_p = relevance / z_p
        c_p = tf.matmul(s_p, tf.transpose(W_p))

        W_n = tf.maximum(0., kernel)
        b_n = tf.maximum(0., bias)
        z_n = tf.matmul(activation, W_n) + b_n
        s_n = relevance / z_n
        c_n = tf.matmul(s_n, tf.transpose(W_n))

        return activation * (self.alpha * c_p + (1 - self.alpha) * c_n)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

logdir = 'dir/'
ckptdir = logdir + 'model'

if not os.path.exists(logdir):
    os.mkdir(logdir)


def training():
    with tf.name_scope('Classifier'):
        # Initialize neural network
        DNN = MNIST_DNN('DNN')

        # Setup training process
        X = tf.placeholder(tf.float32, [None, 784], name='X')
        Y = tf.placeholder(tf.float32, [None, 10], name='Y')

        activations, logits = DNN(X)

        tf.add_to_collection('LRP', X)

        for activation in activations:
            tf.add_to_collection('LRP', activation)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

        optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=DNN.vars)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cost_summary = tf.summary.scalar('Cost', cost)
    accuray_summary = tf.summary.scalar('Accuracy', accuracy)
    summary = tf.summary.merge_all()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    # Hyper parameters
    training_epochs = 4
    batch_size = 100
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        avg_cost = 0
        avg_acc = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c, a, summary_str = sess.run([optimizer, cost, accuracy, summary], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
            avg_acc += a / total_batch

            file_writer.add_summary(summary_str, epoch * total_batch + i)

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =',
              '{:.9f}'.format(avg_acc))

        saver.save(sess, ckptdir)
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    sess.close()


def display_lrp():
    # ensure the network has been trained
    if len(os.listdir(os.path.join(os.getcwd(), logdir))) == 0:
        raise RuntimeError('please train the network firstly before display lrp')

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(logdir))
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*kernel.*')
    biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*bias.*')
    activations = tf.get_collection('LRP')
    X = activations[0]
    lrp = LRP(1, activations, weights, biases, 'LRP')
    Rs = []
    for i in range(10):
        Rs.append(lrp(i))
    images = mnist.train.images
    labels = mnist.train.labels
    sample_imgs = []
    for i in range(10):
        sample_imgs.append(images[np.argmax(labels, axis=1) == i][3])
    imgs = []
    for i in range(10):
        imgs.append(sess.run(Rs[i], feed_dict={X: sample_imgs[i][None, :]}))
    sess.close()
    plt.figure(figsize=(8, 8))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(np.reshape(imgs[2 * i], [28, 28]), cmap='hot_r')
        plt.title('Digit: {}'.format(2 * i))
        plt.colorbar()

        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(np.reshape(imgs[2 * i + 1], [28, 28]), cmap='hot_r')
        plt.title('Digit: {}'.format(2 * i + 1))
        plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    display_lrp()
    # training()
    pass