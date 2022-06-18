from pathlib import Path
from .components.layer import Layer
from .components import FullyConnected, Conv2D, MaxPool2D
from typing import List, Tuple
import numpy as np
import pickle

class Model:
    """
    Representation of CNN model
    """
    def __init__(self, input_size=(28,28,1), layers=(Conv2D(1, 4, 2), MaxPool2D(), FullyConnected(13*13*4, 10)), loss='cross_entropy') -> None:
        self.input_size:Tuple = input_size
        self.input_h = input_size[0]
        self.input_w = input_size[1]
        self.input_c = input_size[2]
        self.layers:List[Layer] = list(layers)
        self.loss = loss
        

    def save(self, path:Path):
        """
        Save model into file
        """
        pickle.dump(self, open(str(path), 'wb'))
    
    @staticmethod
    def load(path):
        """
        Load model from file
        """
        model = pickle.load(open(str(path), 'rb'))
        return  model

    def calculate_grad(self, pred, true_label):
        """
        Output grad calculation
        """
        grad = np.zeros_like(pred)
        grad[true_label] = -1 / pred[true_label]
        return grad



    def forward(self, x, label=None):
        """
        Model forward on given [x] input image. In case of training [label] is needed to calculate loss.
        """
        # check dim of x
        if len(x.shape) == len(self.input_size) - 1:
            # expand x dim into H x W x C
            x = np.expand_dims(x, axis=2)

        # preprocess input
        x = (x / 255) - 0.5
        
        # formard across each layer
        for layer in self.layers:
            x = layer.forward(x)
        # last layer output is prediction 
        pred = x

        if label is not None:
            # loss calculation
            if self.loss == 'cross_entropy':
                loss = -np.log(pred[label])
            else:
                raise Exception(f'Unnown loss function: {self.loss}')
            return pred, loss
        else:
            return pred
        

    def fit_sample(self, x, y, lr=0.001):
        """
        Fit model at one image sample
        """
        # forward
        pred, loss = self.forward(x, y)
        # calculate initial gradient
        grad = self.calculate_grad(pred, y)
        # backpropagation
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        # udate weights by gradients
        for layer in reversed(self.layers):
            layer.update(lr)
        # get prediction
        pred_label = np.argmax(pred)
        return pred_label, loss


    def fit_batch(self, x, y, lr=0.001):
        """
        Fit model at one batch of image samples
        """
        loss = 0
        batch_pred = []
        # predict and calculate gradients
        for x_sample, y_sample in zip(x, y):
            # forward
            pred, loss_sample = self.forward(x_sample, y_sample)
            # calculate initial gradient
            grad = self.calculate_grad(pred, y_sample)
            # backpropagation
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
            # add prediction
            pred_label = np.argmax(pred)
            batch_pred.append(pred_label)
            # add loss
            loss += loss_sample
        # udate weights by accumulated gradients
        for layer in reversed(self.layers):
            layer.update(lr)
        return batch_pred, loss

