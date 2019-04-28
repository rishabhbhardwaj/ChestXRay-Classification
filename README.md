# ChestXRay-Classification
ChestXRay Disease Diagnosis

## About
A Deep Neural Network model for multi-label thorax disease classification on chest X-ray images. 
The dataset,CheXpert, provides High-Resolution and Low-Resolution images labeled with 14 classes.Our classfication problem consists of only 5 classes as mentioned in the CheXpert Competition.  
We experimented with architectures like DenseNet-121, NASNet4, Resnet-152 with similar parameters, activation and loss function.  
AUROC and Precision Recall are used as metric for the evaluation of the models.

## Architecture/Models

Denset-121 -> classifier (sigmoid activation function per class) 


Resnet-151 -> classifier (sigmoid activation function per class)

NAsNetLarge -> classifier (sigmoid activation function per class)

## Dataset
CheXpert: https://arxiv.org/abs/1901.07031

We have included a subset of Dataset in this repo for CSE 6250 Evaluation Purpose.

Dataset can be downloaded from here: http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip

Note: Please note that you need to subscribe to CheXpert website to download the dataset: https://stanfordmlgroup.github.io/competitions/chexpert/

## Models
Model which is submitted to [CheXpert competition](https://stanfordmlgroup.github.io/competitions/chexpert/) is available at dir: ```codalab/src/best_weights_1555982768.7076797.h5```

Currently, **We are placed 4th (fourth) in the competition**.

NASNetLarge model can be downloaded from this [link](https://www.dropbox.com/s/i42s1its6r6nd9n/best_weights_1556219541.2163985_NASNet.h5?dl=0)

## For quick evaluation of the existing models, please follow below steps:
Pre-requisite : Anaconda

Set-up:
```
conda create -n chexpert python=3.6.5
pip install -r requirements_eval.txt
```

Run the model:

```
cd <root directory>
python src/test.py
```

NOTE: Please note that "test.py" runs on a small sample dataset, AOC scores might not be as expected.

## Set Up
```
pip install -r requirements.txt
```

## Training
```
export PYTHONPATH=<Path-To-ChestXRay-Classification>:$PYTHONPATH
mkdir <Path-To-ChestXRay-Classification>/out
python src/train.py
```

## Testing
```
export PYTHONPATH=<Path-To-ChestXRay-Classification>:$PYTHONPATH
python src/test.py --data-dir <Dir Containing CheXpert Data> --model-file-path <Model Weights File Path> --model-type 'DenseNet121'
```

## Results

Below is the model comparison for the 5 classes ('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion')

| Class\Models     	| DenseNet121 	| NASNet-Large 	|
|------------------	|-------------	|--------------	|
| Atelectasis      	| 0.808819    	| 0.805774     	|
| Cardiomegaly     	| 0.823752    	| 0.800245     	|
| Consolidation    	| 0.923713    	| 0.853493     	|
| Edema            	| 0.918155    	| 0.925000     	|
| Pleural Effusion 	| 0.910326    	| 0.932858     	|
| **Mean AUROC**       	| **0.876953**    	| **0.863474**     	|


## Code Structure
```
├── README.md
├── codalab
│   ├── CheXpert-v1.0
│   │   └── valid
│   │       └── patient00000
│   │           ├── study1
│   │           │   ├── view1_frontal.jpg
│   │           │   └── view2_lateral.jpg
│   │           └── study2
│   │               └── view1_frontal.jpg
│   ├── src
│   │   ├── best_weights_1555982768.7076797.h5
│   │   ├── codalab_submit.py
│   │   └── models.py
│   ├── src.zip
│   └── valid_image_paths.csv
├── config.ini
├── notebooks
│   ├── dataPrep.ipynb
│   ├── evaluate.ipynb
│   └── inspect_model.ipynb
├── out
├── requirements.txt
├── src
│   ├── augmentations.py
│   ├── callbacks.py
│   ├── generator.py
│   ├── metrics.py
│   ├── models.py
│   ├── test.py
│   ├── train.py
│   └── utils.py
├── test_imgs
│   ├── view1_frontal.jpg
│   ├── view1_frontal2.jpg
│   └── view1_frontal3.jpg
├── train4.out
├── train5.out
└── weights
    ├── best_weights_1555865398.1238055_Apr22_5cls.h5
    └── best_weights_1555982768.7076797.h5
```
