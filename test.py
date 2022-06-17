from cnn.model import Model
from cnn.components.measure import measure_acc, measure_f1, measure_precision, measure_recall
from data.mnist import get_mnist_data
import numpy as np
from argparse import ArgumentParser
from pathlib import Path


def measure_model(model, x, y):
    """
    Measure model scores by making prediction at [x] and compare to true labels [y]
    """
    loss = 0
    pred = []
    for x_sample, y_sample in zip(x, y):
        pred_sample, loss_sample = model.forward(x_sample, y_sample)
        pred_sample_label = np.argmax(pred_sample)
        loss += loss_sample
        pred.append(pred_sample_label)
        
    acc = measure_acc(pred, y)
    precision = measure_precision(pred, y)
    recall = measure_recall(pred, y)
    f1 = measure_f1(pred, y)
    return dict(loss=loss, acc=acc, precision=precision, recall=recall, f1=f1)

def parse_args():
    """
    Parse test stript argument
    """
    parser = ArgumentParser(
        "Test saved CNN model on test MNIST dataset"
    )
    parser.add_argument('--model_path', type=lambda p: Path(p).absolute(), required=True)
    args = parser.parse_args()
    return args

def test(model_path:Path):
    """
    Test CNN saved model [model_path] score on MNIST test dataset
    """
    # load model
    model = Model.load(model_path)
    # get data
    _, _, x_test, y_test = get_mnist_data(test=True, valid_fraction=None)
    scores = measure_model(model, x_test, y_test)
    # display scores
    print('--- MNIST TEST RESULTS ---')
    for metric_name, metric_score in scores.items():
        print(f'metric: {metric_name:>13}, value: {metric_score}')
    print('---')

if __name__ == '__main__':
    args = parse_args()
    test(**vars(args))