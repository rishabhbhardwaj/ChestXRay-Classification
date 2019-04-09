from keras.applications import DenseNet121
from keras import layers
from keras import models
from keras.layers.core import Dense


def DenseNet(height, width, channels, classes):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(height, width, channels))
    for layer in base_model.layers[-4:]:
        layer.trainable = False
    x = base_model.layers[-1].output  # es la salida del ultimo activation despues del add
    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.GlobalMaxPool2D()(x)
    # output layer
    #---1) NO LINEAL + LINEAL
    # prepredictions = Dense(256, activation='relu')(x)
    # predictions = Dense(classes, activation='sigmoid')(prepredictions)

    #---2) LINEAL
    predictions = Dense(classes, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    return model