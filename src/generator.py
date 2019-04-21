import os
import pandas as pd
from PIL import Image
from skimage.transform import resize

import numpy as np
import keras


class CheXpertDataGenerator(keras.utils.Sequence):
    'Data Generetor for CheXpert'

    def __init__(self, dataset_csv_file, class_names, source_image_dir, batch_size=16,
                 target_size=(224, 224), policy = "zeroes", augmenter=None, verbose=0,
                 shuffle_on_epoch_end=False, random_state=1):

        self.dataset_df = pd.read_csv(dataset_csv_file)
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmenter = augmenter
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.policy = policy
        self.prepare_dataset()
        self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))

        # print('steps..', self.steps)
    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        # print('idx....', idx)
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def load_image(self, image_file):
        image_path = os.path.join(self.source_image_dir, image_file)
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, self.target_size)
        return image_array

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            batch_x = self.augmenter.augment_images(batch_x)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        if batch_x.shape == imagenet_mean.shape:
            batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps*self.batch_size, :]

    def prepare_dataset(self):
        self.dataset_df = self.dataset_df[self.dataset_df['Frontal/Lateral'] == 'Frontal']
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        df.fillna(0, inplace=True)
        self.x_path, y_df = df["Path"].as_matrix(), df[self.class_names]

        self.y = np.empty(y_df.shape, dtype=int)
        # print(y_ar.shape)
        for i, (index, row) in enumerate(y_df.iterrows()):
            labels = []
            for cls in self.class_names:
                #         print(cls)
                curr_val = row[cls]
                #         print(curr_val)
                feat_val = 0
                if curr_val:
                    curr_val = float(curr_val)
                    if curr_val == 1:
                        feat_val = 1
                    elif curr_val == -1:
                        if self.policy == "ones":
                            feat_val = 1
                        elif self.policy == "zeroes":
                            feat_val = 0
                        else:
                            feat_val = 0
                    else:
                        feat_val = 0
                else:
                    feat_val = 0
                labels.append(feat_val)
            # print(labels)
            self.y[i] = labels

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()
