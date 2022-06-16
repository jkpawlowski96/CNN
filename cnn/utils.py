from pathlib import Path
from datetime import datetime

LOC = Path(__file__).parent
DEFAULT_TRAIN_SAVE_LOCATION = LOC.parent / 'TRAIN'

def get_save_location(parent_dir:Path, prefix=None):
    # create parent dir
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True)
    # get name of new directory with datetime
    dir = parent_dir / f'{datetime.now().strftime("%y-%m-%d__%H_%M_%S")}'
    # add prefix after datetime
    if prefix:
        dir = dir.with_name(f'{dir.name}_{prefix}')
    # if dir exists add more details
    if dir.exists():
        dir = dir.with_name(f'{dir.name}_{datetime.now().microsecond}')
    # create dir
    dir.mkdir()
    return dir
    

