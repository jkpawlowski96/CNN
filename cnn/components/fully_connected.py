import numpy as np
from .layer import Layer

class FullyConnected(Layer):
    """
    Fully connected with softmax model output layer
    """
    def __init__(self, input_size, cells):
        self.input_size = input_size
        self.cells = cells

        # init weights
        self.weights = np.random.randn(input_size, cells)/input_size
        # init biases
        self.biases = np.random.randn(cells)
        # init gradients
        self.reset()

    def reset(self):
        """
        Reset weights and biases gradient
        """
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

    
    def forward(self, input):
        """
        Forward fully_connected output at gived [input]
        """
        # keep input shape to backward operation
        self.last_input_shape = input.shape
        # reshape input into single dimension
        input = input.flatten()
        # keep input to backward operation
        self.last_input = input
        # calculate output
        totals = np.dot(input, self.weights) + self.biases
        # keep last totals to backward operation
        self.last_totals = totals
        # softmax output
        exp = np.exp(totals)
        return exp/np.sum(exp, axis=0)
    
    def backward(self, grad):
        """  
        Provide backpropagation on this layer, basing on [grad] which is loss gradient of this layer output.
        Backward returns gradient of layer's input, to be able of calculate previous layer gradient.
        """
        # iterate oved grad to find non zero element
        for i, _grad in enumerate(grad):
            # check grad
            if(_grad == 0):
                # nothing to calculate
                continue

            # grad totals calculation
            t_exp = np.exp(self.last_totals)
            sum = np.sum(t_exp)
            grad_totals = -t_exp[i] * t_exp/ (sum**2)
            grad_totals[i] = t_exp[i] * (sum-t_exp[i]) /(sum**2)

            # initializations of weight, biases and input gradients
            local_grad_weights = self.last_input
            local_grad_biases = 1
            local_grad_inputs = self.weights

            # calculate gradient of total inputs
            grad_input_totals = _grad * grad_totals
            # accumulate gradient of weights
            self.grad_weights += local_grad_weights[np.newaxis].T @ grad_input_totals[np.newaxis]
            # accumulate gratient of biases
            self.grad_biases += grad_input_totals * local_grad_biases  
            # calculate input gradient
            grad_input = local_grad_inputs @ grad_input_totals
            # reshape grad into input size
            return grad_input.reshape(self.last_input_shape)


    def update(self, lr):
        """
        Update layer weights and biases basing on accumulated gradients and [lr] learning rate. 
        After an update gradients are getting reset.
        """
        self.weights -= lr * self.grad_weights
        self.biases -= lr * self.grad_biases
        self.reset()