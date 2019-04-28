import argparse
import os
import pandas as pd

from src.models import ModelFactory
from src.generator import CheXpertDataGenerator
from keras.preprocessing import image
from src.models import DenseNet, ModelFactory
from skimage.transform import resize
import numpy as np
import keras
from sklearn.metrics.ranking import roc_auc_score


def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []
    for i in range(classCount):
        try:
            print(i)
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
        except ValueError:
            pass
    return outAUROC

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing...')
    parser.add_argument('--data-dir', help='Input dataset directory.', type=str, default='.')
    parser.add_argument('--model-file-path', help='Path to model', type=str, default='./codalab/src/best_weights_1555982768.7076797.h5')
    parser.add_argument('--model-type', help='Model architecture to train', type=str, default='DenseNet121')

    args = parser.parse_args()
    print(args.data_dir)
    img_width, img_height = 224, 224
    valid_file = os.path.join(args.data_dir, 'CheXpert-v1.0-small/valid.csv')
    # model_file_path = '../weights/best_weights_1555865398.1238055_Apr22_5cls.h5'
    class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


    model_factory = ModelFactory()
    model = model_factory.get_model( class_names,
                                    model_name=args.model_type,
                                    use_base_weights=True,
                                    weights_path=args.model_file_path,
                                    input_shape=(img_height, img_width, 3))
    optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    valid_data = CheXpertDataGenerator(dataset_csv_file=valid_file,
                class_names=class_names,
                source_image_dir=args.data_dir,
                batch_size=1,
                target_size=(224, 224),
                augmenter=None,
                policy ='mixed',
            )
    print('Evaluating Model...')
    df = pd.read_csv(valid_file)
    df = df[df['Frontal/Lateral'] == 'Frontal']
    gt = []
    pred = []
    for index, row in df.iterrows():
        # if index == 10:
        #     break
        test_image_path = row['Path']
        study_path = test_image_path.split('/')[:-1]
        study_path = "/".join(study_path)
        # print(study_path)
        img = image.load_img(test_image_path, target_size=(img_width, img_height))
        image_array = np.asarray(img.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, (224, 224))
        x = np.expand_dims(image_array, axis=0)
        images = np.vstack([x])
        classes = model.predict_on_batch(images)
        probab = classes[0]
        curr_gt = row[class_names].values
        gt.append(curr_gt)
        pred.append(probab)
    pred = np.array(pred, dtype=float)
    gt = np.array(gt, dtype=float)

    aurocIndividual = computeAUROC(gt, pred, 5)
    aurocMean = np.array(aurocIndividual).mean()
    print ('AUROC mean ', aurocMean)
    for i in range (0, len(aurocIndividual)):
        print (class_names[i], ' ', aurocIndividual[i])





