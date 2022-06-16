from train import train
from src.model import Model
from src.components import Dense, Conv2D, MaxPool2D

def experiment():
    models = [
        # one conv layer
        Model(layers=(
            Conv2D(1, 4, kernel_size=3),
            MaxPool2D(),
            Dense(13*13*4, 10)
            )),
        Model(layers=(
            Conv2D(1, 8, kernel_size=3),
            MaxPool2D(),
            Dense(13*13*8, 10)
        )),
        Model(layers=(
            Conv2D(1,  16, kernel_size=3),
            MaxPool2D(),
            Dense(13*13*16, 10)
            )),
        Model(layers=(
            Conv2D(1,  32, kernel_size=3),
            MaxPool2D(),
            Dense(13*13*32, 10)
        )),
        # two conv layers
        Model(layers=(
            Conv2D(1, 4, kernel_size=3),
            MaxPool2D(),
            Conv2D(4, 8, kernel_size=2),
            MaxPool2D(),
            Dense(5*5*8, 10)
            )),
        Model(layers=(
            Conv2D(1, 8, kernel_size=3),
            MaxPool2D(),
            Conv2D(8, 16, kernel_size=2),
            MaxPool2D(),
            Dense(5*5*16, 10)
        )),
        Model(layers=(
            Conv2D(1,  16, kernel_size=3),
            MaxPool2D(),
            Conv2D(16, 32, kernel_size=2),
            MaxPool2D(),
            Dense(5*5*32, 10)
            )),
        Model(layers=(
            Conv2D(1,  32, kernel_size=3),
            MaxPool2D(),
            Conv2D(32, 64,kernel_size=2),
            MaxPool2D(),
            Dense(5*5*64, 10)
        )),
    ]
    history_list = []
    test_res_list = []
    for model in models:
        model, history, test_res = train(model, epochs=100)
        history_list.append(history)
        test_res_list.append(test_res)
    print(test_res_list)

if __name__ == '__main__':
    experiment()