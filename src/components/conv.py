import numpy as np
from .layer import Layer

class Conv2D(Layer):
    def __init__(self, num_filters, kernel_size=3):
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        #why divide by 9...Xavier initialization
        self.filters = np.random.randn(num_filters, self.kernel_size, self.kernel_size)/9

        self.reset()
    
    def iterate_regions(self, image):
        #generates all possible k * k image regions using valid padding
        
        h,w = image.shape
        
        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+self.kernel_size), j:(j+self.kernel_size)]
                yield im_region, i, j
                
    def forward(self, input):
        self.last_input = input
        
        h,w = input.shape
        
        output = np.zeros((h-2, w-2, self.num_filters))
        
        for im_regions, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_regions * self.filters, axis=(1,2))
        return output
    
    def backward(self, d_l_d_out):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        '''

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                self.d_l_d_filters[f] += d_l_d_out[i,j,f] * im_region


        return None

    
    def update(self, lr):
        #update filters
        self.filters -= lr * self.d_l_d_filters
        self.reset()

    def reset(self):
        self.d_l_d_filters = np.zeros(self.filters.shape)



