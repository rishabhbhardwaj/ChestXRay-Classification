# ChestXRay-Classification
ChestXRay Disease Diagnosis

## About
TODO

## Architecture/Models
Baseline Model:

Denset-121 -> classifier (sigmoid activation function per class)
Resnet-151 -> classifier (sigmoid activation function per class)

## Dataset
CheXpert: https://arxiv.org/abs/1901.07031

## Set Up
```
pip install -r requirements.txt
```

## Training
```
export PYTHONPATH=<Path-To-ChestXRay-Classification>:$PYTHONPATH
python src/train.py
```
## Results
