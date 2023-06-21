import os
import sys
import csv
import pickle

import numpy as np
import matplotlib.pyplot as plt

import absl
from absl import app
from absl import flags
from absl import logging

logging.use_absl_handler()
# logging.get_absl_handler().setFormatter(None)
logging.set_verbosity(logging.INFO)

flags.DEFINE_float('lr', 0.01, 'learning rate.')
flags.DEFINE_float('momentum', 0.5, 'momentum value for optimizer.')
flags.DEFINE_integer('num_hidden', 3, 'number of hidden layers in the neural network.')
flags.DEFINE_integer('epochs', 50, 'number of epochs for training.')
flags.DEFINE_list('sizes', '100,100,50', 'number of hidden units in each hidden layer')
flags.DEFINE_enum('activation', 'sigmoid', ['sigmoid', 'tanh', 'relu'], 'activation function in the neural network.')
flags.DEFINE_enum('loss', 'ce', ['ce', 'sq'], 'loss function to be used.')
flags.DEFINE_enum('opt', 'gd', ['gd', 'momentum'], 'type of optimizer to use.')
flags.DEFINE_integer('batch_size', 1, 'batch size for training.')
flags.DEFINE_bool('anneal', False, 'annealing for learning rate.')
flags.DEFINE_string('save_dir', 'tmp_out', 'directory for saving the pickled model.')
flags.DEFINE_string('expt_dir', 'tmp_out/exp', 'directory for saving the logs for the model.')
flags.DEFINE_string('train', '', 'path to the training dataset.')
flags.DEFINE_string('train_labels', '', 'path to the training labels.')
flags.DEFINE_string('test', '', 'path to the test images.')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

if not os.path.exists(FLAGS.expt_dir):
    os.makedirs(FLAGS.expt_dir)

labels_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
labels = {}
labels['label'] = -1
for idx, lbl in enumerate(labels_list):
    labels[lbl] = idx


def load_data(images_path, labels_path=None):
    image_files = os.listdir(images_path)
    num_files = len(image_files)
    images_load = [plt.imread(images_path + f'/{i}.png') for i in range(1, num_files + 1)]
    x_data = np.array(images_load)
    x_data = np.reshape(x_data, (x_data.shape[0], 32 * 32 * 3))

    if labels_path is not None:
        with open(labels_path, 'r') as rfile:
            csvfile = csv.reader(rfile)
            y_data = []
            for line in csvfile:
                y_data.append(labels[line[-1]])
        rfile.close()
        y_data = np.array(y_data[1:])
        assert x_data.shape[0] == y_data.shape[0]
        return x_data, y_data
    else:
        return x_data


x_train, y_train = load_data(FLAGS.train, FLAGS.train_labels)
x_test = load_data(FLAGS.test)

p = np.random.permutation(x_train.shape[0])
x_train, y_train = x_train[p], y_train[p]
x_val, y_val = x_train[:5000], y_train[:5000]
x_train, y_train = x_train[5000:], y_train[5000:]

logging.info('Loaded dataset')
logging.info(f'Train --- data shape: {x_train.shape}, labels shape: {y_train.shape}')
logging.info(f'Valid --- data shape: {x_val.shape}, labels shape: {y_val.shape}')
logging.info(f'Test --- data shape: {x_test.shape}')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    ex = np.exp(z - np.max(z))
    return ex / np.sum(ex + 1e-12, axis=0)

def initialize_network(input_dims, hidden_dim_list):
    architecture = [input_dims] + hidden_dim_list + [10] # [3072, 100, 10]
    L = len(architecture)
    w, b, vw, vb = {}, {}, {}, {}

    for l in range(1, L):
        w[l] = np.random.randn(architecture[l], architecture[l - 1]) * np.sqrt(6 / (architecture[l] + architecture[l - 1]))
        b[l] = np.zeros((architecture[l], 1))
        vw[l] = np.zeros((architecture[l], architecture[l - 1]))
        vb[l] = np.zeros((architecture[l], 1))

    return w, b, vw, vb

def forward_pass(x_data, w, b, activation):
    L = len(w)
    z = {}
    a = {}
    a[0] = x_data.T

    if activation == 'sigmoid':
        for l in range(1, L):
            z[l] = np.dot(w[l], a[l - 1]) + b[l]
            a[l] = sigmoid(z[l])
    elif activation == 'tanh':
        for l in range(1, L):
            z[l] = np.dot(w[l], a[l - 1]) + b[l]
            a[l] = tanh(z[l])
    elif activation == 'relu':
        for l in range(1, L):
            z[l] = np.dot(w[l], a[l - 1]) + b[l]
            a[l] = relu(z[l])
    z[L] = np.dot(w[L], a[L - 1]) + b[L]
    a[L] = softmax(z[L])

    return z, a

def loss(y_data, a, loss_type='ce'):
    num_samples = y_data.shape[0]
    y_hot = np.eye(10)[y_data]
    L = len(a) - 1
    preds = a[L].T

    assert preds.shape == y_hot.shape

    if loss_type == 'ce':
        preds = np.clip(preds, 1e-10, 1 - 1e-10)
        loss = -np.multiply(y_hot, np.log(preds))
        loss = np.sum(loss) / num_samples
    elif loss_type == 'sq':
        loss = (y_hot - preds) ** 2
        loss = np.sum(loss) / num_samples
    return loss

def accuracy(x_data, y_data, w, b, activation):
    _, a = forward_pass(x_data, w, b, activation)
    num_samples = y_data.shape[0]
    L = len(a) - 1
    preds = a[L].T
    y_pred = np.argmax(preds, axis=-1)
    return (100 / num_samples) * np.sum(y_pred == y_data)

def backward_pass(y_data, z, a, w, b, activation, loss_type='ce'):
    num_samples = y_data.shape[0]
    y_hot = np.eye(10)[y_data]
    grads = {}
    grads['w'], grads['b'], grads['a'], grads['z'] = {}, {}, {}, {}
    L = len(a) - 1
    if loss_type == 'ce':
        grads['a'][L] = a[L] - y_hot.T
    elif loss_type == 'sq':
        grads['a'][L] = 2 * (a[L] - y_hot.T)
    grads['z'][L] = grads['a'][L] * a[L] * (1 - a[L])
    grads['w'][L] = (1 / num_samples) * np.dot(grads['z'][L], a[L - 1].T)
    grads['b'][L] = (1 / num_samples) * np.sum(grads['z'][L], axis=1, keepdims=True)
    grads['a'][L - 1] = np.dot(w[L].T, grads['z'][L])

    if activation == 'sigmoid':
        for l in range(L - 1, 0, -1):
            grads['z'][l] = grads['a'][l] * a[l] * (1 - a[l])
            grads['w'][l] = (1 / num_samples) * np.dot(grads['z'][l], a[l - 1].T)
            grads['b'][l] = (1 / num_samples) * np.sum(grads['z'][l], axis=1, keepdims=True)
            grads['a'][l - 1] = np.dot(w[l].T, grads['z'][l])
    elif activation == 'tanh':
        for l in range(L - 1, 0, -1):
            grads['z'][l] = grads['a'][l] * (1 - a[l] ** 2)
            grads['w'][l] = (1 / num_samples) * np.dot(grads['z'][l], a[l - 1].T)
            grads['b'][l] = (1 / num_samples) * np.sum(grads['z'][l], axis=1, keepdims=True)
            grads['a'][l - 1] = np.dot(w[l].T, grads['z'][l])
    elif activation == 'relu':
        for l in range(L - 1, 0, -1):
            grads['z'][l] = np.array(grads['a'][l], copy = True)
            (grads['z'][l])[z[l] <= 0] = 0
            grads['w'][l] = (1 / num_samples) * np.dot(grads['z'][l], a[l - 1].T)
            grads['b'][l] = (1 / num_samples) * np.sum(grads['z'][l], axis = 1, keepdims = True)
            grads['a'][l - 1] = np.dot(w[l].T, grads['z'][l])
    
    return grads

def update(w, b, vw, vb, grads, lr, mu, opt='gd'):
    if opt == 'gd':
        mu = 0
    for l in range(1, len(w) + 1):
        vw[l] = mu * vw[l] - lr * grads['w'][l]
        w[l] += vw[l]
        vb[l] = mu * vb[l] - lr * grads['b'][l]
        b[l] += vb[l]
    return w, b, vw, vb

def train_model(x_train, y_train, x_val, y_val, hidden_dim_list, lr=0.1, mu=0.5, bs=1, activation='sigmoid', loss_type='ce', opt='momentum', epochs=50, expt_dir=None):
    if expt_dir is not None:
        logging.get_absl_handler().use_absl_log_file('logs.txt', f'./{expt_dir}')
        logging.get_absl_handler().setFormatter(None)
    hidden_dim_list = list(map(int, hidden_dim_list))
    total_samples, dim = x_train.shape[0], x_train.shape[1]
    w, b, vw, vb = initialize_network(input_dims=dim, hidden_dim_list=hidden_dim_list)

    num_steps = int(np.ceil(total_samples / bs))
    curr_step = 1
    for i in range(epochs):
        startidx = 0
        for j in range(1, num_steps + 1):
            endidx = min(startidx + bs, total_samples - 1)
            if endidx > startidx:
                x_train_batch = x_train[startidx:endidx, :]
                y_train_batch = y_train[startidx:endidx]
                z, a = forward_pass(x_train_batch, w, b, activation)
                loss_value = loss(y_train_batch, a, loss_type)
                if curr_step % 100 == 0:
                    verr = np.around(100 - accuracy(x_val, y_val, w, b, activation), 3)
                    terr = np.around(100 - accuracy(x_train, y_train, w, b, activation), 3)
                    logging.info(f'Epoch {i}, Step {curr_step}, Loss: {np.around(loss_value, 4)}, Train Error: {terr}, Valid Error: {verr} lr: {lr}')
                grads = backward_pass(y_train_batch, z, a, w, b, activation)
                w, b, vw, vb = update(w, b, vw, vb, grads, lr, mu, opt)
                startidx += bs
                curr_step += 1

    return w, b

w, b = train_model(x_train, y_train, x_val, y_val, FLAGS.sizes, FLAGS.lr, FLAGS.momentum, FLAGS.batch_size, FLAGS.activation, FLAGS.loss, FLAGS.opt, FLAGS.epochs, FLAGS.expt_dir)

_, a_test_pred = forward_pass(x_test, w, b, FLAGS.activation)
y_test_pred = a_test_pred[len(a_test_pred) - 1].T
y_test_pred = np.argmax(y_test_pred, axis=-1)

with open('submission.csv', 'w') as fw:
    fw.write('id,label\n')
    for idx in range(y_test_pred.shape[0]):
        fw.write(f'{idx},{labels_list[y_test_pred[idx]]}\n')

pdump = open(FLAGS.save_dir + 'model.pkl', 'wb')
pickle.dump((w, b, FLAGS.activation), pdump)
pdump.close()
