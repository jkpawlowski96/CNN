import numpy as np
from .layer import Layer

class Dense(Layer):
    def __init__(self, input_size, cells):
        self.input_size = input_size
        self.cells = cells
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.randn(input_size, cells)/input_size
        self.biases = np.random.randn(cells)
        self.reset()

    def reset(self):
        self.d_l_d_w = np.zeros_like(self.weights)
        self.d_l_d_b = np.zeros_like(self.biases)

    
    def forward(self, input):
        
        self.last_input_shape = input.shape
        
        input = input.flatten()
        self.last_input = input
                
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        
        exp = np.exp(totals)
        return exp/np.sum(exp, axis=0)
    
    def backward(self, d_l_d_out):
        """  
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layers inputs.
        - d_L_d_out is the loss gradient for this layers outputs.
        """
        
        #We know only 1 element of d_l_d_out will be nonzero
        for i, gradient in enumerate(d_l_d_out):
            if(gradient == 0):
                continue
            
            #e^totals
            t_exp = np.exp(self.last_totals)
            
            #Sum of all e^totals
            S = np.sum(t_exp)
            
            #gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp/ (S**2)
            d_out_d_t[i] = t_exp[i] * (S-t_exp[i]) /(S**2)
            
            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
            
            #Gradients of loss against totals
            d_l_d_t = gradient * d_out_d_t
            
            #Gradients of loss against weights/biases/input
            self.d_l_d_w += d_t_d_w[np.newaxis].T @ d_l_d_t[np.newaxis]
            self.d_l_d_b += d_l_d_t * d_t_d_b  
            d_l_d_inputs = d_t_d_inputs @ d_l_d_t
            
            return d_l_d_inputs.reshape(self.last_input_shape)


    def update(self, lr):
        #update weights/biases
        self.weights -= lr * self.d_l_d_w
        self.biases -= lr * self.d_l_d_b
        self.reset()