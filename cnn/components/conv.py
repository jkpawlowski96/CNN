import numpy as np
from .layer import Layer

class Conv2D(Layer):
    """
    Convolution 2D model layer
    """
    def __init__(self, channels, filters, kernel_size=3):
        self.channels = channels
        self.filters = filters
        self.kernel_size = kernel_size
        # init weights and biases
        scale=0.5
        self.weights = np.random.uniform(-scale, scale, size=(self.kernel_size, self.kernel_size, channels, filters))
        self.biases = np.random.uniform(-scale, scale, size=(1, 1, 1, filters))
        # init gradients
        self.reset()
    
    def iterate_regions(self, image):
        """
        Returns iterable object which contains image regions to perform layer operations
        """
        # image shape
        h,w = image.shape[:2]
        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+self.kernel_size), j:(j+self.kernel_size)]
                yield im_region, i, j
                
    def forward(self, input):
        """
        Forward conv2d output at gived [input]
        """
        # keep last input to grad backward calculation
        self.last_input = input
        # input shape
        h, w, c = input.shape
        # init blank output 
        out = np.zeros((h-2, w-2, self.filters))
        # iterate over regions
        for region, i, j in self.iterate_regions(input):
            # iterate over filters
            for f in range(self.filters):
                # calculate and fill output
                out[i, j] += np.sum(np.multiply(region, self.weights[:,:,:,f])) + float(self.biases[:,:,:,f])
        return out
    
    def backward(self, grad):
        """  
        Provide backpropagation on this layer, basing on [grad] which is loss gradient of this layer output.
        Backward returns gradient of layer's input, to be able of calculate previous layer gradient.
        """
        # init input gradient
        grad_input = np.zeros_like(self.last_input)
        # handle padding
        grad = np.pad(grad, ((1,1),(1,1),(0,0)), 'constant', constant_values=(0,0))
        # iterate over regions
        for im_region, i, j in self.iterate_regions(self.last_input):
            # iterate over filters
            for i_f in range(self.filters):
                # calculate input gradient
                d = self.weights[:,:,:,i_f] * grad[i, j, i_f]
                grad_input[i:i+self.kernel_size, j:j+self.kernel_size] += d
                # calculate weights grad
                a = grad[i:i+self.kernel_size, j:j+self.kernel_size, i_f]
                a = np.expand_dims(a, axis=2)
                d = im_region * a
                # accumulate weights grad
                self.grad_weights[:,:,:, i_f] += d
                # accumulate biases grad
                self.grad_biases[:,:,:,i_f] += grad[i,j,i_f]
        return grad_input

    
    def update(self, lr):
        """
        Update layer weights and biases basing on accumulated gradients and [lr] learning rate. 
        After an update gradients are getting reset.
        """
        self.weights -= lr * self.grad_weights
        self.biases -= lr * self.grad_biases
        self.reset()

    def reset(self):
        """
        Reset weights and biases gradient
        """
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)




