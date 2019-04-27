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

NAsNet4 -> classifier (sigmoid activation function per class)

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

## Testing
```
export PYTHONPATH=<Path-To-ChestXRay-Classification>:$PYTHONPATH
python src/test.py --data-dir <Dir Containing CheXpert Data> --model-file-path <Model Weights File Path> --model-type 'DenseNet121'
```

## Results
| Class\Models     	| DenseNet121 	| NASNet-Large 	|
|------------------	|-------------	|--------------	|
| Atelectasis      	| 0.808819    	| 0.805774     	|
| Cardiomegaly     	| 0.823752    	| 0.800245     	|
| Consolidation    	| 0.923713    	| 0.853493     	|
| Edema            	| 0.918155    	| 0.925000     	|
| Pleural Effusion 	| 0.910326    	| 0.932858     	|
| Mean AUROC       	| 0.876953    	| 0.863474     	|