class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, d, lr):
        raise NotImplementedError