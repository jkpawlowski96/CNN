from termios import OFDEL
from .components.layer import Layer
from .components import Dense, Conv2D, MaxPool2D
from typing import List, Tuple
import numpy as np
import pickle

class Model:
    def __init__(self, input_size=(28,28,1), layers=(Conv2D(1, 4, 2), MaxPool2D(), Dense(13*13*4, 10)), loss='cross_entropy') -> None:
        self.inpus_size:Tuple = input_size
        self.inpus_h = input_size[0]
        self.input_w = input_size[1]
        self.input_c = input_size[2]
        self.layers:List[Layer] = list(layers)
        self.loss = loss
        pass

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
    
    @staticmethod
    def load(path):
        model = pickle.load(open(path, 'rb'))
        return  model

    def calculate_grad(self, pred, true_label):
        grad = np.zeros_like(pred)
        grad[true_label] = -1 / pred[true_label]
        return grad



    def forward(self, x, label):
        if len(x.shape) == len(self.inpus_size) - 1:
            x = np.expand_dims(x, axis=2)
        x = (x / 255) - 0.5
        
        for layer in self.layers:
            x = layer.forward(x)    
        pred = x
        if self.loss == 'cross_entropy':
            loss = -np.log(pred[label])

        elif self.loss == 'mse':
            true = np.zeros_like(pred)
            true[label] = 1
            loss = np.sum([pow(p-t,2) for p,t in zip(pred, true)])/len(pred)
        else:
            raise Exception(f'Unnown loss function: {self.loss}')

        return pred, loss


    def fit_sample(self, x, y, lr=0.001):
        #forward
        pred, loss = self.forward(x, y)
        
        # calculate initial gradient
        grad = self.calculate_grad(pred, y)

        
        # Backpropagation
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            layer.update(lr)
        pred_label = np.argmax(pred)
        return pred_label, loss



    def fit_batch(self, x, y, lr=0.001):
        loss = 0
        batch_pred = []
        for x_sample, y_sample in zip(x, y):
            #forward
            pred, loss_sample = self.forward(x_sample, y_sample)
            # calculate initial gradient
            grad = self.calculate_grad(pred, y_sample)
            # Backpropagation
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
            pred_label = np.argmax(pred)
            batch_pred.append(pred_label)
            loss += loss_sample

        # Update weights by accumulated gradients
        for layer in reversed(self.layers):
            layer.update(lr)
        
        return batch_pred, loss

