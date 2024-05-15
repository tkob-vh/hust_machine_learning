import numpy as np
import matplotlib.pyplot as plt
import functools
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score


class LinearLayer:
    '''
    This class represents a linear layer with ReLU fuction in a neural network.
    '''
    def __init__(self, input_size, output_size):
        '''
        input_size: int, size of input
        output_size: int, size of output
        '''
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size) # weights initialization
        self.weights_grads = np.empty_like(self.weights) # gradient of weights
        self.biases = np.random.randn(output_size) / 10. # biases initialization
        self.biases_grads = np.empty_like(self.biases) # gradient of biases

        self.activation_function = self.ReLU
        self.activation_function_gradient = self.ReLU_backward

    def forward(self, x):
        self.x = x
        self.value = np.matmul(x, self.weights) + self.biases # linear transformation
        self.activation_value = self.activation_function(self.value) #  activation function 
        return self.activation_value
    
    def backward(self, output_grad):
        activation_grad = self.activation_function_gradient(output_grad) # gradient of activation function
        self.biases_grads = activation_grad.reshape(self.biases_grads.shape) # gradient of biases
        self.weights_grads = np.matmul(self.x.T, activation_grad) # gradient of weights 
        return np.matmul(activation_grad, self.weights.T) # gradient of input
    
    def step(self, learning_rate=1e-3):
        self.weights -= learning_rate * self.weights_grads
        self.biases -= learning_rate * self.biases_grads

    def ReLU(self, x):
        return np.maximum(x, 0)
    
    def ReLU_backward(self, output_grad):
        return output_grad * (self.activation_value > 0).astype(float)


class MLP:
    def __init__(self, input_size, output_size, n_layers, layer_dims):
        layer_dims = [input_size] + layer_dims + [output_size]

        self.layers = [LinearLayer(layer_dims[i], layer_dims[i + 1]) for i in range(n_layers)] # list of layers in the network

        self.layers[-1].activation_function = lambda x: x
        self.layers[-1].activation_function_gradient = lambda x: x

        self.loss_function = self.softmax_crossentropy
        self.loss_function_grad = self.softmax_crossentropy_grad

        self.metrics = defaultdict(list)

        self.trained = False

    def forward(self, x):
        layers = [functools.partial(layer.forward) for layer in self.layers]
        return functools.reduce(lambda x, y: y(x), layers, x)
    
    def backward(self, output_grad):
        layer_grads = [functools.partial(layer.backward) for layer in reversed(self.layers)]
        return functools.reduce(lambda x, y: y(x), layer_grads, output_grad)

    def softmax_crossentropy(self, y, ygt):
        y = y - np.max(y, axis=1, keepdims=True)  # 防止数值溢出
        exp_y = np.exp(y)
        softmax = exp_y / (np.sum(exp_y, axis=1, keepdims=True) + 1e-6)
        return -np.log(softmax[np.arange(y.shape[0]), ygt] + 1e-6)


    def softmax_crossentropy_grad(self, y, ygt):
        softmax = np.exp(y) / (np.exp(y).sum() + 1e-6)
        softmax[0, ygt] -= 1
        return softmax
    
    def step(self, learning_rate):
        for layer in self.layers:
            layer.step(learning_rate)
    
    def fit(self, X_train, y_train, learning_rate=1e-3, n_epochs=20, show_progress=True):
        train_size = X_train.shape[0]

        for epoch in range(1, n_epochs + 1):
            y_pred = []
            running_loss = []

            for sample in range(train_size):
                x = X_train[sample].reshape((1, -1))

                ygt_i = np.array([y_train[sample]])
                y_i = self.forward(x)

                loss = self.loss_function(y_i, ygt_i)
                loss_grad = self.loss_function_grad(y_i, ygt_i)
                running_loss.append(loss)

                self.backward(loss_grad)
                self.step(learning_rate)
                y_pred.extend(y_i.argmax(1))

            self.metrics['accuracy'].append(accuracy_score(y_train, y_pred))
            self.metrics['balanced accuracy'].append(balanced_accuracy_score(y_train, y_pred))
            self.metrics['recall'].append(precision_score(y_train, y_pred, average='micro'))
            self.metrics['precision'].append(precision_score(y_train, y_pred, average='micro'))
            self.metrics['loss'].append(np.mean(running_loss))

            if show_progress == True and epoch % 1 == 0:
                 print(f'Epoch: {epoch}/{n_epochs}\tloss: {np.mean(running_loss):.3f}\t',\
                      f'balanced accuracy on train: {balanced_accuracy_score(y_train, y_pred):.3f}')
                 
        self.trained = True

    def predict(self, X_test):
        if self.trained == False:
            raise ValueError('Model not trained yet')
        y_pred = []
        for sample in X_test:
            X_i = sample.reshape((1, -1))
            y_i = self.forward(X_i)
            y_pred.extend(y_i.argmax(1))
        return np.array(y_pred)


def train_mlp(X_train, y_train):
    mlp = MLP(input_size=X_train.shape[1], output_size=10, n_layers=3, layer_dims=[128, 128, 64])
    mlp.fit(X_train, y_train, learning_rate=1e-3, n_epochs=20, show_progress=True)
    return mlp

def inference_mlp(X_test, mlp):
    return mlp.predict(X_test)