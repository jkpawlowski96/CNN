import imp
from cnn.components.measure import measure_acc, measure_f1, measure_precision, measure_recall
import numpy as np


def measure_model(model, x, y):
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


