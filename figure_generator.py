import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from core50 import *


def load_precached_folds(args, seed=42):
    '''
    Load the data set from the arguments.

    The underlying files represent individual folds.  Several are combined together to
    create the training set.

    The Core 50 data set has:
    - 10 object classes
    - 5 object instances per class
    - 11 background conditions in which each of the object instances are imaged in

    :param args: Command line arguments
    :param seed: Random seed

    :return: TF Datasets for the training, validation and testing sets + number of classes
    '''

    # Open meta data file to pull out the number of objects
    fname = args['dataset'] + '/' + args['precache'] + '/meta.csv'
    df = pd.read_csv(fname)
    nobjects = df['nclasses'][0]
    nfolds = df['nfolds'][0]

    # Test create object-based rotations
    core = Core50(args['dataset'] + '/' + args['meta_dataset'])

    # Check to make sure that argument matches that actual number of folds
    assert nfolds == args['Nfolds'], "args.Nfolds does not match actual number of folds"

    # Load the fold-wise data sets
    ds_folds = []
    for i in range(args['Nfolds']):
        name = 'core50_objects_10_fold_%d' % i

        # Create the foldwise datasets
        ds = tf.data.Dataset.load('%s/%s/%s' % (args['dataset'], args['precache'], name))
        ds_folds.append(ds)

    # Create training/validation/test DFs
    # This step does the batching/prefetching/repeating/shuffling
    ds_training, ds_validation, ds_testing = core.create_training_validation_testing_from_datasets(args['rotation'],
                                                                                                   ds_folds,
                                                                                                   args['Ntraining'],
                                                                                                   cache=args['cache'],
                                                                                                   batch_size=args['batch'],
                                                                                                   prefetch=args['prefetch'],
                                                                                                   repeat=args['repeat'],
                                                                                                   shuffle=args['shuffle'])
    # Done
    return ds_training, ds_validation, ds_testing, nobjects


def figure3():
    deep_test_accuracy = np.empty(5)
    shallow_test_accuracy = np.empty(5)
    for r in range(5):
        with open(f'results/image_Csize_5_3_Cfilters_10_10_Pool_2_2_Pad_valid_hidden_50_20_LR_0.001000_ntrain_03_rot_'
                  f'{r:02d}_results.pkl', "rb") as fp:
            results = pickle.load(fp)
            shallow_test_accuracy[r] = results['predict_testing_eval'][1]
        with open(f'results/image_Csize_1_5_3_1_5_3_1_5_3_1_5_3_1_5_3_Cfilters_8_8_8_16_16_16_32_32_32_64_64_64_32_32_32'
                  f'_Pool_1_1_1_2_1_1_2_1_1_2_1_1_2_1_1_Pad_same_hidden_1024_512_256_128_64_drop_0.500_sdrop_0.200_'
                  f'L2_0.001000_LR_0.000100_ntrain_03_rot_{r:02d}_results.pkl', "rb") as fp:
            results = pickle.load(fp)
            deep_test_accuracy[r] = results['predict_testing_eval'][1]
    fig = plt.figure()
    plt.hist(shallow_test_accuracy, bins=5, color='blue', edgecolor='black', alpha=0.5, label='Shallow')
    plt.hist(deep_test_accuracy, bins=5, color='red', edgecolor='black', alpha=0.5, label='Deep')
    plt.title('Frequency of Test Accuracy')
    plt.xlabel('Test Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    fig.savefig('figures/fig3.png')


def figure4(args):
    ds_train, ds_validation, ds_testing, n_classes = load_precached_folds(args)
    model = keras.models.load_model('image_Csize_5_3_Cfilters_10_10_Pool_2_2_Pad_valid_hidden_50_20_LR_0.001000_'
                                    'ntrain_03_rot_00_model')
    for ins, outs in ds_testing.take(1):
        model.predict(ins)
        fig = plt.figure()
        plt.imshow(ins)
        plt.title('test')
        fig.savefig('images/test.png')


if __name__ == '__main__':
    args = {
        'dataset': '/scratch/fagg/core50',
        'precache': 'datasets_by_fold_4_objects',
        'meta_dataset': 'core50_df.pkl',
        'Nfolds': 5,
        'rotation': 0,
        'Ntraining': 3,
        'cache': '',
        'batch': 1,
        'prefetch': 8,
        'repeat': False,
        'shuffle': 0
    }
    figure4(args)
