from keras.models import load_model
from keras.preprocessing import image
from src.models import DenseNet, ModelFactory
import numpy as np
import keras
import sys
import pandas as pd

img_width, img_height = 224, 224
class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
model_type = "DenseNet121"

def load_model():
    model_file_path = '/Users/rishabhbhardwaj/Downloads/best_weights_1555739761.3928604.h5'
    model_factory = ModelFactory()
    model = model_factory.get_model( class_names,
                                    model_name=model_type,
                                    use_base_weights=False,
                                    weights_path=model_file_path,
                                    input_shape=(img_height, img_width, 3))
    optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "binary_accuracy"])

    return model

def predict_save(model, input_file, output_file):

    test_df = pd.read_csv(input_file)
    for index, row in test_df.iterrows():
        test_image_path = row['Path']
        img = image.load_img(test_image_path, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict_on_batch(images)
        #TODO Convert in Codalab submission format (per study line)


if __name__ == '__main__':
    input_filename = sys.argv[1]
    output_csv = sys.argv[2]
    model = load_model()
    predict_save(model, input_filename, output_csv)