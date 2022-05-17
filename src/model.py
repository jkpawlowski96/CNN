from .components.layer import Layer
from .components import Dense, Conv2D, MaxPool2D
from typing import List, Tuple
import numpy as np

class Model:
    def __init__(self, input_size=(28,28,1), layers=(Conv2D(4, 2), MaxPool2D(), Dense(13*13*4, 10))) -> None:
        self.inpus_size:Tuple = input_size
        self.inpus_h = input_size[0]
        self.input_w = input_size[1]
        self.input_c = input_size[2]
        self.layers:List[Layer] = list(layers)
        pass


    def forward(self, x, label):
        x = (x / 255) - 0.5
        for layer in self.layers:
            x = layer.forward(x)    
        
        #calculate cross-entropy loss and accuracy
        loss = -np.log(x[label])
        return x, loss


    def fit_sample(self, x, y, lr=0.001):
        #forward
        pred, loss = self.forward(x, y)
        
        # calculate initial gradient
        grad = np.zeros(10)
        grad[y] = -1/pred[y]
        
        # Backpropagation
        for layer in reversed(self.layers):
            gradient = layer.backward(grad)
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
            grad = np.zeros(10)
            grad[y_sample] = -1/pred[y_sample]
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

