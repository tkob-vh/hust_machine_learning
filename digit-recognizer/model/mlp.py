import numpy as np
import matplotlib.pyplot as plt
import functools
import os
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
        self.biases = np.zeros(output_size) # biases initialization
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
        '''
        Initialize the MLP model
        '''
        layer_dims = [input_size] + layer_dims + [output_size]

        self.layers = [LinearLayer(layer_dims[i], layer_dims[i + 1]) for i in range(n_layers)] # list of layers in the network

        self.layers[-1].activation_function = lambda x: x
        self.layers[-1].activation_function_gradient = lambda x: x

        self.loss_function = self.softmax_crossentropy
        self.loss_function_grad = self.softmax_crossentropy_grad

        self.metrics = defaultdict(list)

        self.trained = False

    def forward(self, x):
        '''
        Forward pass through the network
        '''
        layers = [functools.partial(layer.forward) for layer in self.layers]
        return functools.reduce(lambda x, y: y(x), layers, x)
    
    def backward(self, output_grad):
        '''
        Backward pass through the network
        '''
        layer_grads = [functools.partial(layer.backward) for layer in reversed(self.layers)]
        return functools.reduce(lambda x, y: y(x), layer_grads, output_grad)

    def softmax_crossentropy(self, y, ygt):
        '''
        Compute the softmax crossentropy loss
        '''
        y = y - np.max(y, axis=1, keepdims=True)  # 防止数值溢出
        exp_y = np.exp(y)
        softmax = exp_y / (np.sum(exp_y, axis=1, keepdims=True) + 1e-6)
        return -np.log(softmax[np.arange(y.shape[0]), ygt] + 1e-6)


    def softmax_crossentropy_grad(self, y, ygt):
        '''
        Compute the gradient of the softmax crossentropy loss
        '''
        softmax = np.exp(y) / (np.exp(y).sum() + 1e-6)
        softmax[0, ygt] -= 1
        return softmax
    
    def step(self, learning_rate):
        '''
        Update the weights and biases of the network
        '''
        for layer in self.layers:
            layer.step(learning_rate)
    
    def fit(self, X_train, y_train, learning_rate=1e-3, n_epochs=20, show_progress=True):
        '''
        Train the model
        '''
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
            self.metrics['recall'].append(recall_score(y_train, y_pred, average='micro'))
            self.metrics['precision'].append(precision_score(y_train, y_pred, average='micro'))
            self.metrics['loss'].append(np.mean(running_loss))

            if show_progress:
                balanced_acc = balanced_accuracy_score(y_train, y_pred)
                acc = accuracy_score(y_train, y_pred)
                recalls = recall_score(y_train, y_pred, average=None)
                precisions = precision_score(y_train, y_pred, average=None)
                
                print(f'Epoch: {epoch}/{n_epochs}\tloss: {np.mean(running_loss):.3f}')
                print(f'balanced accuracy on train: {balanced_acc:.3f}')
                print(f'accuracy on train: {acc:.3f}')
                
                for i, (rec, prec) in enumerate(zip(recalls, precisions)):
                    print(f'class {i} - recall: {rec:.3f}, precision: {prec:.3f}')

                 
        self.trained = True

    def predict(self, X_test):
        '''
        Predict the output for the test data
        '''
        if self.trained == False:
            raise ValueError('Model not trained yet')
        y_pred = []
        for sample in X_test:
            X_i = sample.reshape((1, -1))
            y_i = self.forward(X_i)
            y_pred.extend(y_i.argmax(1))
        return np.array(y_pred)


def train_mlp(X_train, y_train):
    '''
    Train the MLP model
    '''
    mlp = MLP(input_size=X_train.shape[1], output_size=10, n_layers=3, layer_dims=[128, 128, 64])
    mlp.fit(X_train, y_train, learning_rate=1e-3, n_epochs=20, show_progress=True)
    return mlp

def inference_mlp(X_test, mlp):
    '''Inference the MLP model on the test data
    '''
    return mlp.predict(X_test)

def plot(mlp):
    n_metrics = len(mlp.metrics)
  
    for i, metric in enumerate(mlp.metrics):
        n_epochs = len(mlp.metrics[metric])
        fig, ax = plt.subplots(figsize=(10, 5))  # Create a separate figure for each metric

        ax.plot(range(1, n_epochs + 1), mlp.metrics[metric], color='k')
        ax.axhline(mlp.metrics[metric][-1], linestyle='--', color='g')
        ax.set_title(metric)
        ax.set_xlabel('epoch')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Save the plot as a separate image
        filename = os.path.join('log/mlp', f'{metric}.png')  # Generate filename based on metric
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to avoid memory leaks
