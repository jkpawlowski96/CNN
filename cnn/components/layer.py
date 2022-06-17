from abc import abstractmethod


class Layer:
    """
    Base class of layer component
    """
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update(self, lr):
        pass

    def reset(self):
        pass