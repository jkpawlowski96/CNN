# CNN
Intelligent computation techniques - master degree project

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --filters 4,8 --skip 500 --epochs 200
```

## Experiment

```bash
python experiment.py
```

## Test

```bash
python test.py --model_path model.pckl
```

## Predict

```bash
python predict.py --source image.jpg --model_path model.pckl
```

## Sources
- https://kaifabi.github.io/2020/01/15/numpy_mlp.html
