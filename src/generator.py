import os
import pandas as pd
from PIL import Image

import numpy as np
import keras


class CheXpertDataGenerator(keras.utils.Sequence):
    'Data Generetor for CheXpert'

    def __init__(self, train_file, classes, data_dir, batch_size=32, dim=(390 * 320), n_channels=1,
                 shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.classes = classes
        self.n_classes = len(self.classes)
        self.dim = dim
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.policy = 'ones'

        self.train_df = pd.read_csv(train_file)
        self.train_df = self.train_df[self.train_df['Frontal/Lateral'] == 'Frontal']
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.train_df.shape[0]) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        curr_batch = self.train_df.iloc[indexes]
        X, y = self.__data_generation(curr_batch)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.train_df.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, curr_batch):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for index, row in curr_batch.iterrows():
            # Image
            img = Image.open(os.path.join(self.data_dir, row['Path']))
            img = img.resize(self.dim, Image.ANTIALIAS)
            X.append(np.array(img))

            # Label
            labels = []
            for cls in self.classes:
                curr_val = row[cls]
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

            y.append(labels)

        return X, y