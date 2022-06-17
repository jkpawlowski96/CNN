from sys import prefix
from train import train
from cnn.model import Model
from cnn.components import FullyConnected, Conv2D, MaxPool2D


def experiment(**kwargs):
    """
    The experiment consists of experimenting with training many models \
    of different configurations with the same assumed learning parameters
    """
    # one conv layer
    train(Model(layers=(Conv2D(1, 4, kernel_size=3),
                        MaxPool2D(),
                        FullyConnected(13*13*4, 10))), **kwargs, prefix='conv_4')
    train(Model(layers=(
        Conv2D(1, 8, kernel_size=3),
        MaxPool2D(),
        FullyConnected(13*13*8, 10)
    )), **kwargs, prefix='conv_8')
    train(Model(layers=(
        Conv2D(1,  16, kernel_size=3),
        MaxPool2D(),
        FullyConnected(13*13*16, 10)
    )), **kwargs, prefix='conv_16')

    train(Model(layers=(
        Conv2D(1,  32, kernel_size=3),
        MaxPool2D(),
        FullyConnected(13*13*32, 10)
    )), **kwargs, prefix='conv_32')

    # two conv layers
    train(Model(layers=(
        Conv2D(1, 4, kernel_size=3),
        MaxPool2D(),
        Conv2D(4, 8, kernel_size=2),
        MaxPool2D(),
        FullyConnected(5*5*8, 10)
    )), **kwargs, prefix='conv_4_8')
    train(Model(layers=(
        Conv2D(1, 8, kernel_size=3),
        MaxPool2D(),
        Conv2D(8, 16, kernel_size=2),
        MaxPool2D(),
        FullyConnected(5*5*16, 10)
    )), **kwargs, prefix='conv_8_16')
    train(Model(layers=(
        Conv2D(1,  16, kernel_size=3),
        MaxPool2D(),
        Conv2D(16, 32, kernel_size=2),
        MaxPool2D(),
        FullyConnected(5*5*32, 10)
    )), **kwargs, prefix='conv_16_32')
    train(Model(layers=(
        Conv2D(1,  32, kernel_size=3),
        MaxPool2D(),
        Conv2D(32, 64, kernel_size=2),
        MaxPool2D(),
        FullyConnected(5*5*64, 10)
    )), **kwargs, prefix='conv_32_64')


if __name__ == '__main__':
    experiment(epochs=100, patience=None, skip=100)
