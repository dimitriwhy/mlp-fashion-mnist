import numpy as np
from activation import *

class FullyConnected:
    def __init__(self, dim_in, dim_out, batch_size, activation):
        #initialization according to He et al.(2015)
        self.W = np.random.randn(dim_out, dim_in).astype(np.float32) * np.sqrt(2.0/(dim_in))
        self.b = np.zeros([dim_out]).astype(np.float32)
        self.batch_size = batch_size
        self.activation = activation

    def forward(self, ipnuts):
        self.inputs = inputs
        outputs = np.dot(inputs, self.W) + self.b
        self.outputs = outputs
        outputs = activation.forward(outputs)
        self.outputs_act = outputs
        return outputs

    def backward(self, grad):
        activ_grad = activation.backward(inputs, outputs_act, grad)
        self.grad_b = activ_grad
        self.grad_W = np.dot(self.inputs.transpose(), activ_grad)
        grad_inputs = np.dot(self.W, activ_grad.tranpose()).transpose()
        return grad_inputs

    def update(self, lr):
        self.W = self.W - lr * np.mean(self.grad_W)
        self.b = self.b - lr * np.mean(self.grad_b)
