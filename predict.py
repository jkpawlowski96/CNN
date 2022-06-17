from cnn.model import Model
from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np

def parse_args():
    parser = ArgumentParser(
        "Predict MNIST label by CNN model"
    )
    parser.add_argument('--model_path', required=True, type=lambda p: Path(p).absolute())
    parser.add_argument('--source', required=True, type=lambda p: Path(p).absolute())
    parser.add_argument('--format', required=False, default="jpg")
    return parser.parse_args()

def predict(model_path:Path, source:Path, format:str):
    """
    Make prediction of CNN model for given files in [source]
    """
    model = Model.load(str(model_path))
    print('model loaded')
    if source.is_dir():
        files = source.glob(f'*.{format}')
    else:
        files = source
    for i, file in enumerate(files):
        # read image
        img = cv2.imread(file)
        # convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize image
        img = cv2.resize(img, (model.input_w, model.input_h))
        # predict
        pred = model.forward(img)
        label = np.argmax(pred)
        # logging
        print(f'-[{i}]- file: {file}, label: {label}')


if __name__ == '__main__':
    args = parse_args()
    predict(**vars(args))


