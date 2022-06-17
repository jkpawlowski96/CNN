from cnn.model import Model
from argparse import ArgumentParser
from pathlib import Path

def parse_args():
    parser = ArgumentParser(
        "Predict MNIST label by CNN model"
    )
    parser.add_argument('--model_path', required=True, type=lambda p: Path(p).absolute())