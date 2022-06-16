import numpy as np
from .layer import Layer

class Conv2D(Layer):
    def __init__(self, channels, filters, kernel_size=3):
        self.channels = channels
        self.filters = filters
        self.kernel_size = kernel_size
        scale=0.5
        self.weights = np.random.uniform(-scale, scale, size=(self.kernel_size, self.kernel_size, channels, filters))
        self.bias = np.random.uniform(-scale, scale, size=(1, 1, 1, filters))
        self.reset()
    
    def iterate_regions(self, image):
        h,w = image.shape[:2]
        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+self.kernel_size), j:(j+self.kernel_size)]
                yield im_region, i, j
                
    def forward(self, input):
        self.last_input = input
        
        h, w, c = input.shape
        out = np.zeros((h-2, w-2, self.filters))
        
        for region, i, j in self.iterate_regions(input):
            for f in range(self.filters):
                out[i, j] += np.sum(np.multiply(region, self.weights[:,:,:,f])) + float(self.bias[:,:,:,f])
        return out
    
    def backward(self, d_l_d_out):
        x = self.last_input
        dx = np.zeros_like(self.last_input)
        d_l_d_out = np.pad(d_l_d_out, ((1,1),(1,1),(0,0)), 'constant', constant_values=(0,0))
        for im_region, i, j in self.iterate_regions(x):
            for i_f in range(self.filters):
                tmp = self.weights[:,:,:,i_f] * d_l_d_out[i, j, i_f]
                dx[i:i+self.kernel_size, j:j+self.kernel_size] += tmp

                a = d_l_d_out[i:i+self.kernel_size, j:j+self.kernel_size, i_f]
                a = np.expand_dims(a, axis=2)
                tmp = im_region * a
                self.d_l_d_weights[:,:,:, i_f] += tmp
                self.d_l_d_bias[:,:,:,i_f] += d_l_d_out[i,j,i_f]
        return dx

    
    def update(self, lr):
        self.weights -= lr * self.d_l_d_weights
        self.bias -= lr * self.d_l_d_bias
        self.reset()

    def reset(self):
        self.d_l_d_weights = np.zeros_like(self.weights)
        self.d_l_d_bias = np.zeros_like(self.bias)




