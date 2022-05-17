import numpy as np
from .layer import Layer

class MaxPool2D(Layer):
    """
    Max pool 2D model layer to reduce input size and training parameters
    """
    def iterate_regions(self, image):
        """
        Returns iterable object which contains image regions to perform layer operations
        """
        # size
        h, w, _ = image.shape
        # new size
        new_h = h // 2
        new_w = w // 2
        # iteration
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j
                
    def forward(self, input):
        """
        MaxPool2D forward at given [input]
        """
        # keep input for backward use
        self.last_input = input
        # input shape
        h, w, f = input.shape
        # empty output
        output = np.zeros((h//2, w//2, f))
        #  iterate over regions and apply max operation
        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.amax(im_region,axis=(0,1))
        return output
    
    def backward(self, grad):
        '''
        Provide backpropagation on this layer
        '''
        # grad of input
        grad_input = np.zeros(self.last_input.shape)

        # iterate over regions of input
        for im_region, i, j in self.iterate_regions(self.last_input):
            # region shape
            h, w, f = im_region.shape
            # max pixel coord
            amax = np.amax(im_region, axis=(0,1))
            # iterate over pixels in region
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # find max pixel
                        if(im_region[i2,j2,f2] == amax[f2]):
                            # apply grad
                            grad_input[i*2+i2, j*2+j2 ,f2] = grad[i, j, f2]
                            break
        return grad_input

    