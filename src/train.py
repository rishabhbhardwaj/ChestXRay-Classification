import argparse
import os
import time
import pickle
from configparser import ConfigParser

import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import models
from src.generator import CheXpertDataGenerator
from src.callbacks import MultipleClassAUROC
from src.utils import get_sample_counts, get_class_weights

def parse_args(args):

    parser = argparse.ArgumentParser(description='Training...')
    parser.add_argument('--data-dir', help='Input dataset directory.', type=str, default='../')
    parser.add_argument('--data-csv-dir', help='path of the folder that contains train.csv|dev.csv|test.csv', type=str, default='../CheXpert-v1.0-small/')
    parser.add_argument('--out-dir', help='Output directory', type=str, default='../data/results/')
    parser.add_argument('--base-model', help='Initial pretrained model.Default is Imagenet.', type=str, default='imagenet')
    parser.add_argument('--model', help='Model architecture to train', type=str, default='densenet121')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=5)
    parser.add_argument('--lr', help='Initial learning rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', help='Training batch size', type=int, default=64)


def get_tf_session():

    config = tf.ConfigProto()
    config.gpu.options.allow_growth = True
    session = tf.Session(config = config)
    return session

def main(args):

    args = parse_args(args)
    config = ConfigParser()
    config.read('../config.ini')

    HEIGHT = 390
    WIDTH = 320
    channels = 3
    output_weights_path = os.path.join(args.out_dir, 'weights_'+str(time.time())+'.h5')
    training_stats = {}
    generator_workers = config["TRAIN"].getint("generator_workers")
    min_lr = config["TRAIN"].getfloat("minimum_lr")
    patience_reduce_lr = config['TRAIN'].getint('patience_reduce_lr')
    positive_weights_multiply = config['TRAIN'].getint('positive_weights_multiply')

    log_dir = os.path.join(args.out_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity' ,
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                   'Pleural Effusion', 'Pleural Other', 'Fracture','Support Devices']

    train_file = os.path.join(args.data_csv_dir, 'train.csv')
    valid_file = os.path.join(args.data_csv_dir, 'valid.csv')

    train_data = CheXpertDataGenerator(train_file, class_names, args.data_dir,batch_size=args.batch_size)
    valid_data = CheXpertDataGenerator(valid_file, class_names, args.data_dir, batch_size=args.batch_size)

    train_counts, train_pos_counts = get_sample_counts(args.data_dir, "train", class_names)
    valid_counts, _ = get_sample_counts(args.data_dir, "valid", class_names)
    class_weights = get_class_weights(
        train_counts,
        train_pos_counts,
        multiply=positive_weights_multiply,
    )

    train_steps = int(train_counts / args.batch_size)
    valid_steps = int(valid_counts / args.batch_size)

    if args.model == 'densenet121':
        model = models.Densenet121(HEIGHT, WIDTH, channels, class_names)
    else:
        print('Model',args.model, 'is not supported.')

    optimizer = keras.optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    auroc = MultipleClassAUROC(
        sequence=valid_data,
        class_names=class_names,
        weights_path=output_weights_path,
        stats=training_stats,
        workers=generator_workers,
    )
    checkpoint = ModelCheckpoint(
        output_weights_path,
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )
    callbacks = [
        checkpoint,
        TensorBoard(log_dir=os.path.join(args, "logs"), batch_size=args.batch_size),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr,
                          verbose=1, mode="min", min_lr=min_lr),
        auroc,
    ]

    print("----------- Training ------------")
    history = model.fit_generator(
        generator=train_data,
        steps_per_epoch=train_steps,
        epochs=args.epochs,
        validation_data=valid_data,
        validation_steps=valid_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=generator_workers,
        shuffle=False,
    )

    print("--------- Store History ----------------")
    with open(os.path.join(args.out_dir, "history.pkl"), "wb") as f:
        pickle.dump({
            "history": history.history,
            "auroc": auroc.aurocs,
        }, f)
    print("--------- Completed ---------")

