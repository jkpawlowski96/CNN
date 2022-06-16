from pathlib import Path
import pickle
from data.utils import limit_float
import pandas as pd


class History:
    def __init__(
            self, 
            prefix_list=['', 'val'],
            measure_list=['loss', 'acc', 'precision', 'recall', 'f1']) -> None:
        # initialize results dictionary
        res = {}
        for prefix in prefix_list:
            for measure in measure_list:
                if prefix:
                    res[f'{prefix}_{measure}'] = []
                else:
                    res[f'{measure}'] = []
        self.res = res

    def get_logging_line(self, key_contain=None, key_not_contain=None):
        keys = list(self.res.keys())
        if key_contain:
            keys = [k for k in keys if key_contain in k]
        if key_not_contain:
            keys = [k for k in keys if key_not_contain not in k]
            
        line = ' '.join([f'{k:>13} {limit_float(self.res[k][-1])}' for k in keys])
        return line

    def add(self, key, value):
        self.res[key].append(value)

    def get_measure(self, key:str, epoch:int):
        if key not in self.res:
            return None
        if epoch > len(self.res[key]) - 1:
            return None
        return self.res[key][epoch]

    def get_dataframe(self) -> pd.DataFrame:
        measures = list(self.res.keys())
        columns = ['epoch'] + measures
        epochs = max([len(m) for m in self.res.values()])
        data = []
        for epoch in range(epochs):
            data_row = {
                'epoch': epoch
            }
            for measure in measures:
                data_row[measure] = self.get_measure(measure, epoch)
            data.append(data_row.copy())
        df = pd.DataFrame(data=data)
        return df

    def save(self, filepath:Path, csv=True):
        """
        Save history in file
        """
        pickle.dump(self.res, open(str(filepath), 'wb'))
        if csv:
            df = self.get_dataframe()
            df.to_csv(filepath.with_suffix('.csv'))

    def load(self, filepath:Path):
        """
        Load history from file
        """
        self.res = pickle.load(open(str(filepath), 'rb'))