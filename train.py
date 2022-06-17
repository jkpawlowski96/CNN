from cnn.components.fully_connected import FullyConnected
from cnn.components.pool import MaxPool2D
from cnn.components.conv import Conv2D
from cnn.model import Model
from data.mnist import get_mnist_data
from cnn.components.measure import measure_acc, measure_precision, measure_recall, measure_f1
from tqdm import tqdm
from test import measure_model
from cnn.utils import DEFAULT_TRAIN_SAVE_LOCATION, get_save_location
from cnn.history import History
from cnn.early_stopping import EarlyStopping
from cnn.checkpoint import Checkpoint
import pandas as pd
from argparse import ArgumentParser

def train(  
        model, 
        epochs=50, 
        lr=0.001, 
        batch_size=4, 
        seed=39571592,
        skip=None,
        valid_fraction=0.15,
        valid=True,  
        test=True,
        patience=None,
        use_checkpoint=True,
        save_model=True,
        save_history=True,
        save_test=True,
        save_location=None,
        prefix=None,
        **kwargs):
    """
    Train CNN model
    """

    # get data
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        valid_fraction=valid_fraction, test=True, skip=skip, seed=seed)

    # measures prefix
    prefix_list = ['']
    if valid:
        prefix_list.append('val')

    if not save_location and (
        save_model or save_history or save_test or use_checkpoint
    ):
        save_location =  get_save_location(DEFAULT_TRAIN_SAVE_LOCATION, prefix=prefix) if not save_location else save_location

    # init history
    history = History(prefix_list=prefix_list)

    if patience:
        # init early_stopping
        early_stopping = EarlyStopping(
            history=history, 
            patience=patience, 
            metric='val_f1' if valid else 'f1', 
            inverse=False)

    if use_checkpoint:
        checkpoint = Checkpoint(
            model=model, 
            history=history, 
            savedir=save_location, 
            metric= 'val_f1' if valid else 'f1',
            inverse=False)

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
            if len(x_batch) == batch_size or i == len(x_train) - 1:
                pred_batch, loss_batch = model.fit_batch(
                    x_batch, y_batch, lr=lr)
                train_loss += loss_batch
                pred += pred_batch
                x_batch = []
                y_batch = []

        # measure training scores
        train_acc = measure_acc(pred, y_train)
        train_precision = measure_precision(pred, y_train)
        train_recall = measure_recall(pred, y_train)
        train_f1 = measure_f1(pred, y_train)
        history.add('loss', train_loss)
        history.add('acc', train_acc)
        history.add('precision', train_precision)
        history.add('recall', train_recall)
        history.add('f1', train_f1)

        # measure validate scores
        if valid:
            valid_measures = measure_model(model, x_valid, y_valid)
            for k, v in valid_measures.items():
                history.add(f'val_{k}', v)

        # logging train
        
        print(history.get_logging_line(key_not_contain='val'))

        # logging val
        if valid:
            print(history.get_logging_line(key_contain='val'))

        if use_checkpoint:
            checkpoint.step()

        if patience:
            if not early_stopping.step():
                print('--- early stopping: BREAK ---')
                break

    if use_checkpoint:
        model = checkpoint.get_best_model()
        print('Checkpoint: got best model')

    # primary results
    results = [model, history]

    # testing
    if test:
        test_scores = measure_model(model, x_test, y_test)
        results.append(test_scores)
        if save_test:
            save_location =  get_save_location(DEFAULT_TRAIN_SAVE_LOCATION, prefix=prefix) if not save_location else save_location
            test_scores_df = pd.DataFrame([test_scores])
            test_scores_df.to_csv(save_location / 'test.csv')

    # store model on drive
    if save_model:
        model.save(save_location / 'model.pckl')
    
    # store training history on drive
    if save_history:
        save_location =  get_save_location(DEFAULT_TRAIN_SAVE_LOCATION, prefix=prefix) if not save_location else save_location
        history.save(save_location / 'history.pckl')

    return results

def parse_args():
    """
    Parse train script argument
    """
    parser = ArgumentParser(
        "Train CNN model"
    )
    parser.add_argument("--filters", 
        required=False, 
        default=[8], 
        type=lambda x: x.split(','),
        help='List of filters numbers \n\
            Examples: \n\
            \'8\' [conv_1 (8 filters)] \n\
            \'4,8\' [conv_1 (4 filters), conv_2 (8 filters)]'
        )
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--valid_fraction', default=0.15)
    parser.add_argument('--batch-size', default=4)
    parser.add_argument('--prefix', required=False, default=None)
    parser.add_argument('--patience', required=False, default=None)
    parser.add_argument('--skip', required=False, default=None, help="Use only [skip] number of samples for training from MNIST dataset")
    args = parser.parse_args()
    return args

def create_model(filters:list):
    """
    Create model object based on filter list
    """
    channels = 1
    layers = []
    size = 28
    for i, _filters in enumerate(filters):
        k = 2 if i == 0 else 3
        layers.append(Conv2D(channels=channels, filters=_filters, kernel_size=k))
        layers.append(MaxPool2D())
        channels = _filters
        size = (size - 2) // 2
    layers.append(FullyConnected(input_size=size*size*_filters, cells=10))
    return Model(layers=layers)

if __name__ == '__main__':
    args = parse_args()
    model = create_model(args.filters)
    train(model=model, **vars(args))
