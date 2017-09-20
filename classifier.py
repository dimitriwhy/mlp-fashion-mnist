import numpy as np
from utils import mnist_reader
import argparse
import logging
from layers import FullyConnected
from activation import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('mlp-fashion-mnist')
np.random.seed(1234)

def evaluate(inputs, y):
    for layer in FCs:
        outputs = layer.forward(inputs)
        outputs = np.argmax(outputs, axis=1)
    precision = float(np.mean(outputs == y))
    return precision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="numpy mlp for fasion-mnist.")
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=200, help='batch size for training.')
    parser.add_argument('--epochs', dest='epochs', type=int, default=25,
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', dest='lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    opts = parser.parse_args()
    
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    X_train = X_train.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255
    r = np.random.permutation(len(y_train))
    X_train = X_train[r]
    y_train = y_train[r]
    X_dev = X_train[:10000]
    y_dev = y_train[:10000]
    X_train = X_train[10000:]
    y_train = y_train[10000:]
    
    LOG.info("finish data preprocessing.")

    FCs = [FullyConnected(784, 256, opts.batch_size, relu()),
           FullyConnected(256, 128, opts.batch_size, relu()),
           FullyConnected(128, 64, opts.batch_size, softmax())]
    
    LOG.info("finish initialization.")

    n_samples = len(y_train)
    order = np.arange(n_samples)
    best_precison , test_precision = 0, 0
    for epochs in range(0, opts.epochs):
        np.random.shuffle(order)
        cost = 0.
        for batch_start in range(0, n_samples, opts.batch_size):
            batch_end = batch_start + opts.batch_size if batch_start \
                        + opts.batch_size < n_samples else n_samples
            batch_id = order[batch_start: batch_end]
            xs, ys = X_train[batch_id], y_train[batch_id]x
            inputs = xs
            for layer in FCs:
                outputs = layer.forward(inputs)
            cost += - np.sum(np.log(outputs) * np.eye(10)[ys])
            grad = -1 / outputs * np.eye(10)[ys]
            for layer in range(2, -1, -1):
                grad = FCs[layer].backward(grad)
            for layer in FCs:
                FCs.update(opts.lr)

        precision = evaluate(X_dev, y_dev)
        LOG.info("iteration {0}, cost = {1}, dev_precision = {2}".format(
            epochs, cost, precision))
        if precision > best_precision:
            best_precision = precision
            test_precision = evaluate(X_test, y_test)
            LOG.info("New best achived. test_precision = {0}"
                     .format(test_precision))
    LOG.info("Training finished. dev_precision = {0}, test_precision = {1}"
             .format(best_precision, test_precision))
