from argparse import ArgumentParser
from pathlib import Path
from typing import List
import pandas as pd
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

ANALYZE_DIRECTORY_NAME = 'analyze_results'

def parse_args():
    parser = ArgumentParser(
        "Analyze training results"
    )
    parser.add_argument('--directory', required=True, type=lambda p: Path(p).absolute())
    return parser.parse_args()

def analyze(directory:Path):
    """
    Analyze results of model training processes
    """
    # find all dirs in location
    dirs = [d for d in list(directory.glob('*')) if d.is_dir() and d.stem is not ANALYZE_DIRECTORY_NAME]
    # get all test results file
    test_files = [d/'test.csv' for d in dirs]
    test_files = filter(lambda f: f.exists(), test_files)
    # get all training history files
    history_files = [d/'history.csv' for d in dirs]
    history_files = filter(lambda f: f.exists(), history_files)
    # create empty analyze results directory
    results_directory = directory / ANALYZE_DIRECTORY_NAME
    if results_directory.exists():
        shutil.rmtree(str(results_directory))
    results_directory.mkdir(parents=True)
    # analyze test results
    analyze_test(test_files, results_directory)
    # analyze train history
    analyze_history(history_files, results_directory)

def join_csv_file(files:List[Path]):
    """
    Join all csv training result files in list into one dataframe
    """
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df['directory'] = file.parent.stem
        df['model'] = file.parent.stem[19:] if len(file.parent.stem) > 19 else None
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    return df

def analyze_test(files:List[Path], results_directory:Path):
    """
    Analyze test results of each training
    """
    df = join_csv_file(files)
    df = df.sort_values('acc', ascending=False)
    df.to_csv(results_directory/'test.csv')

def analyze_history(files:List[Path], results_directory:Path):
    """
    Analyze train history of each training
    """
    # join all files
    df = join_csv_file(files)
    sns.set_theme(style="darkgrid")
    # iterate over measures
    for measure in ['loss', 'acc', 'precision', 'recall' ,'f1']:
        # comparison of all models
        # iterate over subsets
        for prefix in [None, 'val']:

            if prefix:
                _measure = f'{prefix}_{measure}'
                img_path = results_directory / f'all_{_measure}.png'
            else:
                _measure = measure
                img_path = results_directory / f'all_train_{measure}.png'
            # plot data
            sns.lineplot(x="epoch",
                        y=_measure,
                        hue="model", 
                        #style="event",
                        data=df).set(
                            title=img_path.stem.replace('_', ' ').capitalize(),
                            yticks=np.round(np.linspace(
                                np.min(df[_measure].values), 
                                np.max(df[_measure].values), 
                                8), 2)
                        )
            # save plot into image file
            plt.savefig(str(img_path))
            plt.close()
        
        # train vs val for each trained model
        for model_directory in df['directory'].unique():
            # only one model data
            _df = df.loc[df['directory']==model_directory]
            model = _df['model'].values[0]
            if not model:
                model = model_directory
            # prepare data
            plot_df = pd.DataFrame(data={
                'epoch': list(_df['epoch'].values) + list(_df['epoch'].values),
                measure: list(_df[measure]) + list(_df['val_'+measure]),
                'subset': ['train' for _ in range(len(_df))] + ['val' for _ in range(len(_df))]
            })
            # plot
            sns.lineplot(x="epoch",
                            y=measure,
                            hue="subset", 
                            #style="event",
                            data=plot_df).set(
                                title=f'{measure} of {model}'.capitalize(),
                                yticks=np.round(np.linspace(
                                    np.min(df[_measure].values), 
                                    np.max(df[_measure].values), 
                                    8), 2)
                            )
            # save plot into image file
            img_path = results_directory / f'{model}_{measure}'
            plt.savefig(str(img_path))
            plt.close()
            

if __name__ == '__main__':
    args = parse_args()
    analyze(**vars(args))


