import numpy as np

def arrays_inputs(func):
    def f(pred, true, *args, **kwargs):
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)
        if not isinstance(true, np.ndarray):
            true = np.array(true)
        return func(pred, true, *args, **kwargs)
    return f

@arrays_inputs
def measure_acc(pred, true):
    return np.sum((np.round(pred) == true) * 1) / len(true)

@arrays_inputs
def measure_recall(pred, true):
    labels = set(true)
    tp = 0
    fn = 0

    for label in labels:
        tp += np.sum((pred[pred==label] == true[pred==label]) * 1)
        fn += np.sum((pred[pred!=label] != true[pred!=label]) * 1)
    return tp / (tp + fn)

@arrays_inputs
def measure_precision(pred, true):
    labels = set(true)
    tp = 0
    fp = 0

    for label in labels:
        tp += np.sum((pred[pred==label] == true[pred==label]) * 1)
        fp += np.sum((pred[pred==label] != true[pred==label]) * 1)
    return tp / (tp + fp)

@arrays_inputs
def measure_f1(pred, true):
    recall = measure_recall(pred, true)
    precision = measure_precision(pred, true)
    f1 = 2 * (recall * precision) / (precision + recall)
    return f1

