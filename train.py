from statistics import mode
from sys import prefix
from src.model import Model
from data.mnist import get_mnist_data
from data.utils import limit_float
from src.components.measure import measure_acc, measure_precision, measure_recall, measure_f1
from tqdm import tqdm
from test import measure_model


def get_empty_history(
        prefix_list=['', 'val'], 
        measure_list=['loss', 'acc' ,'precision','recall','f1']):
    res = {}
    for prefix in prefix_list:
        for measure in measure_list:
            if prefix:
                res[f'{prefix}_{measure}'] = []
            else:
                res[f'{measure}'] = []
    return res


def train(epochs=50, lr=0.01, batch_size=4, valid=True, seed=39571592):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(valid_fraction=0.15, test=True, skip=1000, seed=seed)
    model = Model()
    prefix_list = ['']
    if valid:
        prefix_list.append('val')

    history = get_empty_history(prefix_list=prefix_list)
    # training process
    for epoch in range(epochs):
        print(f'---- EPOCH {epoch} ----')

        train_loss = 0
        pred = []
        x_batch = []
        y_batch = []
        for i, (x_sample, y_sample) in tqdm(enumerate(zip(x_train, y_train)), total=len(x_train)):
            x_batch.append(x_sample)
            y_batch.append(y_sample)
            if len(x_batch) == batch_size or i == len(x_train) - 1 :
                pred_batch, loss_batch = model.fit_batch(x_batch, y_batch, lr=lr)
                train_loss += loss_batch
                pred += pred_batch
                x_batch = []
                y_batch = []
            
        train_acc = measure_acc(pred, y_train)
        train_precision = measure_precision(pred, y_train)
        train_recall = measure_recall(pred, y_train)
        train_f1 = measure_f1(pred, y_train)
        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['precision'].append(train_precision)
        history['recall'].append(train_recall)
        history['f1'].append(train_f1)
        if valid:
            valid_measures = measure_model(model, x_valid, y_valid)
            for k, v in valid_measures.items():
                history[f'val_{k}'].append(v)
        train_line = ' '.join([f'{k:>13} {limit_float(v[-1])}' for k, v in history.items() if 'val' not in k])
        print(train_line)
        if valid:
            valid_line = ' '.join([f'{k:>13} {limit_float(v[-1])}' for k, v in history.items() if 'val' in k])
            print(valid_line)
    return history

if __name__ == '__main__':
    history = train()
    print(history)