import mnist
from .utils import shuffle as shuffle_arrays
 

def get_mnist_data(valid_fraction=0.15, test=True, shuffle=True, skip=None, seed=None):
    """
    Get mnist dataset devided into subsets
    """
    x_train = mnist.train_images()
    y_train = mnist.train_labels()
    if shuffle:
        x_train, y_train = shuffle_arrays(x_train, y_train, seed=seed)
    if skip:
        x_train, y_train = x_train[:skip], y_train[:skip]
    res = []
    if valid_fraction:
        valid_size = max(int(len(x_train) * valid_fraction),1)
        x_valid = x_train[:valid_size]
        y_valid = y_train[:valid_size]
        x_train = x_train[valid_size:]
        y_train = y_train[valid_size:]
        res += [x_train, y_train, x_valid, y_valid]
    else:
        res += [x_train, y_train]
    if test:
        x_test = mnist.test_images()
        y_test = mnist.test_labels()
        if skip:
            x_test = x_test[:skip]
            y_test = y_test[:skip]
        res.append(x_test)
        res.append(y_test)
    return tuple(res)

