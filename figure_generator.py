import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from core50 import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    # Load accuracies from results files
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

    # Plot histogram of accuracies
    fig = plt.figure()
    bins = np.linspace(start=min(np.min(shallow_test_accuracy), np.min(deep_test_accuracy)),
                       stop=max(np.max(shallow_test_accuracy), np.max(deep_test_accuracy)), num=8)
    plt.hist(shallow_test_accuracy, bins=bins, color='blue', edgecolor='black', alpha=0.5, label='Shallow')
    plt.hist(deep_test_accuracy, bins=bins, color='red', edgecolor='black', alpha=0.5, label='Deep')
    plt.title('Frequency of Test Accuracy')
    plt.xlabel('Test Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    fig.savefig('figures/fig3.png')


def figure4_5(args, num_fig4):
    # Loop over every rotation and generate confusion matrices
    for r in range(5):
        args['rotation'] = r

        # Load dataset and corresponding models
        ds_train, ds_validation, ds_testing, n_classes = load_precached_folds(args)
        shallow_model = keras.models.load_model(f'results/image_Csize_5_3_Cfilters_10_10_Pool_2_2_Pad_valid_hidden_'
                                                f'50_20_LR_0.001000_ntrain_03_rot_{r:02d}_model')
        deep_model = keras.models.load_model(f'results/image_Csize_1_5_3_1_5_3_1_5_3_1_5_3_1_5_3_Cfilters_8_8_8_16_16_'
                                             f'16_32_32_32_64_64_64_32_32_32_Pool_1_1_1_2_1_1_2_1_1_2_1_1_2_1_1_Pad_'
                                             f'same_hidden_1024_512_256_128_64_drop_0.500_sdrop_0.200_L2_0.001000_LR_'
                                             f'0.000100_ntrain_03_rot_{r:02d}_model')

        # Generate figure 4 for only first rotation
        if r == 0:
            # Get first batch
            for ins, outs in ds_testing.take(1):
                # Make predictions with both models
                shallow_predictions = shallow_model.predict(ins)
                deep_predictions = deep_model.predict(ins)

                # Display predictions overlaying images for given number of images
                for i in range(num_fig4):
                    # Show image
                    fig = plt.figure(figsize=(5, 5))
                    plt.imshow(ins[i])
                    plt.axis('off')

                    # Overlay shallow predictions
                    plt.text(0.25, 0.8, 'Shallow', transform=plt.gcf().transFigure, color="black", fontsize=12,
                             ha='center', bbox=dict(facecolor='white', alpha=0.5))
                    for j, text in enumerate(shallow_predictions[i]):
                        plt.text(0.25, 0.8 - (j + 1) * 0.1, f'{j}: {text * 100:.2f}%', transform=plt.gcf().transFigure,
                                 color="black", fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))

                    # Overlay deep predictions
                    plt.text(0.75, 0.8, 'Deep', transform=plt.gcf().transFigure, color="black", fontsize=12,
                             ha='center', bbox=dict(facecolor='white', alpha=0.5))
                    for j, text in enumerate(deep_predictions[i]):
                        plt.text(0.75, 0.8 - (j + 1) * 0.1, f'{j}: {text * 100:.2f}%', transform=plt.gcf().transFigure,
                                 color="black", fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))

                    # Save figure
                    fig.savefig(f'figures/fig4_{i}.png', bbox_inches='tight', pad_inches=0)

        # Loop over test set and find predictions
        shallow_pred_classes = []
        deep_pred_classes = []
        true_classes = []
        for ins, outs, in ds_testing:
            # Make predictions
            shallow_predictions = shallow_model.predict(ins)
            deep_predictions = deep_model.predict(ins)

            # Find prediction with highest softmax activation
            shallow_highest_class = np.empty(shallow_predictions.shape[0])
            deep_highest_class = np.empty(deep_predictions.shape[0])
            for i in range(shallow_predictions.shape[0]):
                shallow_highest_class[i] = np.argmax(shallow_predictions[i])
                deep_highest_class[i] = np.argmax(deep_predictions[i])

            # Add batch to current lists
            shallow_pred_classes.extend(shallow_highest_class)
            deep_pred_classes.extend(deep_highest_class)
            true_classes.extend(outs)

        # Convert python lists to numpy
        shallow_pred_classes = np.array(shallow_pred_classes)
        deep_pred_classes = np.array(deep_pred_classes)
        true_classes = np.array(true_classes)

        # Make confusion matrices
        shallow_cm = confusion_matrix(true_classes, shallow_pred_classes)
        deep_cm = confusion_matrix(true_classes, deep_pred_classes)

        # Make shallow confusion matrix figure
        shallow_disp = ConfusionMatrixDisplay(confusion_matrix=shallow_cm)
        shallow_disp.plot()
        plt.title(f'Shallow Confusion Matrix Rotation {r}')
        plt.savefig(f'figures/fig5_shallow_rot_{r}.png')
        plt.close()

        # Make deep confusion matrix figure
        deep_disp = ConfusionMatrixDisplay(confusion_matrix=deep_cm)
        deep_disp.plot()
        plt.title(f'Deep Confusion Matrix Rotation {r}')
        plt.savefig(f'figures/fig5_deep_rot_{r}.png')
        plt.close()


if __name__ == '__main__':
    args = {
        'dataset': '/scratch/fagg/core50',
        'precache': 'datasets_by_fold_4_objects',
        'meta_dataset': 'core50_df.pkl',
        'Nfolds': 5,
        'Ntraining': 3,
        'batch': 1024,
        'cache': '',
        'prefetch': 8,
        'repeat': False,
        'shuffle': 0
    }
    figure3()
    figure4_5(args, 5)
