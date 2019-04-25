from keras.models import load_model
from keras.preprocessing import image
#from src.models import DenseNet, ModelFactory
from models import DenseNet, ModelFactory
import numpy as np
import keras
import sys
import pandas as pd

img_width, img_height = 224, 224
class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
model_type = "DenseNet121"

def load_model():
    model_file_path = 'src/best_weights_1555982768.7076797.h5'
    #model_file_path = 'best_weights_1555982768.7076797.h5'
    model_factory = ModelFactory()
    model = model_factory.get_model( class_names,
                                    model_name=model_type,
                                    use_base_weights=False,
                                    weights_path=model_file_path,
                                    input_shape=(img_height, img_width, 3))
    optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "binary_accuracy"])
    model.load_weights(model_file_path)

    return model

def predict_save(model, input_file, output_df):

    test_df = pd.read_csv(input_file)
    for index, row in test_df.iterrows():
        test_image_path = row['Path']
        study_path = test_image_path.split('/')[:-1] 
        study_path = "/".join(study_path)
        print(study_path)
        img = image.load_img(test_image_path, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict_on_batch(images)
        probab = classes[0]
        output_df = output_df.append({'Study':study_path, 'Atelectasis':probab[0], 'Cardiomegaly':probab[1], 'Consolidation':probab[2], 'Edema':probab[3], 'Pleural Effusion':probab[4]}, ignore_index=True)
    return output_df

if __name__ == '__main__':
    input_filename = sys.argv[1]
    output_csv = sys.argv[2]
    model = load_model()
    output_df = pd.DataFrame(columns=['Study','Atelectasis','Cardiomegaly','Consolidation','Edema','Pleural Effusion'])
    output_df = predict_save(model, input_filename, output_df)
    output_df = output_df.groupby(['Study']).mean()
    output_df.to_csv(output_csv, sep=',')#, header=False)#, index=False, header=False)
