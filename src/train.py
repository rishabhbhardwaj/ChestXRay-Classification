import argparse
import os
import time
import pickle
from configparser import ConfigParser

import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.backend.tensorflow_backend import set_session

from src.models import DenseNet, ModelFactory
from src.generator import CheXpertDataGenerator
from src.callbacks import MultipleClassAUROC
from src.utils import get_sample_counts, get_class_weights
from src.augmentations import augmenter

def parse_args(args):

    parser = argparse.ArgumentParser(description='Training...')
    parser.add_argument('--data-dir', help='Input dataset directory.', type=str, default='./')
    parser.add_argument('--data-csv-dir', help='path of the folder that contains train.csv|dev.csv|test.csv', type=str, default='./CheXpert-v1.0-small/')
    parser.add_argument('--out-dir', help='Output directory', type=str, default='./out/')
    parser.add_argument('--base-model', help='Initial pretrained model.Default is Imagenet.', type=str, default='imagenet')
    parser.add_argument('--model', help='Model architecture to train', type=str, default='DenseNet121')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=5)
    parser.add_argument('--lr', help='Initial learning rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', help='Training batch size', type=int, default=64)
    parser.add_argument('--weights', help='Pre-trained weights', type=str, default=None)

    return parser.parse_args(args)

def get_tf_session():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    return session

def main(args=None):

    args = parse_args(args)
    cfg = ConfigParser()
    cfg.read('config.ini')
    print("Complete reading config...")

    HEIGHT = 224
    WIDTH = 224
    image_dimension = 224

    current_run_stamp = str(time.time())
    output_weights_path = os.path.join(args.out_dir, 'weights_'+current_run_stamp+'.h5')
    training_stats = {}
    generator_workers = cfg["TRAIN"].getint("generator_workers")
    min_lr = cfg["TRAIN"].getfloat("minimum_lr")
    patience_reduce_lr = cfg['TRAIN'].getint('patience_reduce_lr')
    positive_weights_multiply = cfg['TRAIN'].getint('positive_weights_multiply')
    image_source_dir = cfg["DEFAULT"].get("image_source_dir")

    log_dir = os.path.join(args.out_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity' ,
    #                'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    #                'Pleural Effusion', 'Pleural Other', 'Fracture','Support Devices']

    class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    set_session(get_tf_session())

    train_file = os.path.join(args.data_csv_dir, 'train.csv')
    valid_file = os.path.join(args.data_csv_dir, 'valid.csv')

    train_counts, train_pos_counts = get_sample_counts(train_file, class_names)
    valid_counts, _ = get_sample_counts(valid_file, class_names)
    class_weights = get_class_weights(
        train_counts,
        train_pos_counts,
        multiply=positive_weights_multiply,
    )
    # print('batch size', args.batch_size)
    # print('train counts', train_counts)
    # print('valid counts', valid_counts)
    # train_steps = int(train_counts / args.batch_size)
    # valid_steps = int(valid_counts / args.batch_size)

    train_data = CheXpertDataGenerator(dataset_csv_file=train_file,
            class_names=class_names,
            source_image_dir=image_source_dir,
            batch_size=args.batch_size,
            target_size=(image_dimension, image_dimension),
            augmenter=augmenter,
            policy = 'mixed',
        )
    valid_data = CheXpertDataGenerator(dataset_csv_file=valid_file,
            class_names=class_names,
            source_image_dir=image_source_dir,
            batch_size=args.batch_size,
            target_size=(image_dimension, image_dimension),
            augmenter=None,
            policy ='mixed',
        )

    use_base_model_weights = True
    if args.weights:
        weights_path_file = args.weights
    else:
        weights_path_file = None

    model_factory = ModelFactory()
    if args.model == 'DenseNet121':
        # model = DenseNet(HEIGHT, WIDTH, channels, class_names)
        model = model_factory.get_model(
                                class_names,
                                model_name=args.model,
                                use_base_weights=use_base_model_weights,
                                weights_path=weights_path_file,
                                input_shape=(HEIGHT, WIDTH, 3))
    else:
        print('Model',args.model, 'is not supported.')
        exit(1)

    optimizer = keras.optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "binary_accuracy"])

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
        TensorBoard(log_dir=os.path.join(args.out_dir, "logs"), batch_size=args.batch_size),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr,
                          verbose=1, mode="min", min_lr=min_lr),
        auroc,
    ]

    print("----------- Training ------------")
    history = model.fit_generator(
        generator=train_data,
        steps_per_epoch=None,
        epochs=args.epochs,
        validation_data=valid_data,
        validation_steps=None,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=generator_workers,
        shuffle=False,
    )

    print("--------- Store Current Run ----------------")
    with open(os.path.join(args.out_dir, current_run_stamp+'.pkl'), 'wb') as f:
        pickle.dump({
            'history': history.history,
            'auroc': auroc.aurocs,
        }, f)
    print("--------- Completed ---------")

if __name__ == '__main__':
    main()