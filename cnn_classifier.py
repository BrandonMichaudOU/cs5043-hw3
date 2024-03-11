import tensorflow as tf
from keras.layers import (InputLayer, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D,
                          GlobalMaxPooling2D, SpatialDropout2D)
from tensorflow.keras.models import Sequential


def create_cnn_classifier_network(image_size, nchannels, conv_layers=None, dense_layers=None, p_dropout=None,
                                  p_spatial_dropout=None, lambda_l2=None, lrate=0.001, n_classes=4,
                                  loss='sparse_categorical_crossentropy', metrics=None, padding='valid',
                                  conv_activation='elu', dense_activation='elu'):
    '''
    Creates a sequential convolutional neural network used for image classification
    :param image_size: The size of the input images (2-tuple)
    :param nchannels: Number of color channels of input images
    :param conv_layers: List of dictionaries containing keys for filters, kernel_size, pool_size, strides, and
                        batch_regularization. Each dictionary represents a single convolutional layer followed by a
                        single max pooling layer specified by the key-value pairs.
    :param dense_layers: List of dictionaries containing keys for units and batch_normalization. Each dictionary
                         represents a single dense layer followed by a single batch normalization layer (if True)
    :param p_dropout: Probability of dropout for dense layers
    :param p_spatial_dropout: Probability of spatial dropout for convolutional layers
    :param lambda_l2: L2 regularization hyperparameter. Applies to both convolutional and dense layers
    :param lrate: Learning rate
    :param n_classes: Number of output classes
    :param loss: Loss to optimize
    :param metrics: Additional metrics to calculate
    :param padding: Type of padding to use for convolutions. Can be same or valid
    :param conv_activation: The activation function for the convolutional layers
    :param dense_activation: The activation function for the dense layers
    '''
    # Build sequential model
    model = Sequential()

    # Input units
    model.add(InputLayer(input_shape=(image_size[0], image_size[1], nchannels), name='input'))

    # Convolution modules
    for i, conv_layer in enumerate(conv_layers):
        model.add(Conv2D(filters=conv_layer['filters'], kernel_size=conv_layer['kernel_size'], strides=1,
                         padding=padding, use_bias=True, name=f'conv_{i}', activation=conv_activation,
                         kernel_regularizer=tf.keras.regularizers.l2(l2=lambda_l2)))
        model.add(MaxPooling2D(pool_size=conv_layer['pool_size'], strides=conv_layer['strides'], padding=padding,
                               name=f'max_pool_{i}'))
        model.add(SpatialDropout2D(p_spatial_dropout if p_spatial_dropout is not None else 0))
        if conv_layer['batch_normalization']:
            model.add(BatchNormalization())

    # Global max pooling
    model.add(GlobalMaxPooling2D())

    # Dense modules
    for i, dense_layer in enumerate(dense_layers):
        model.add(Dense(dense_layer['units'], use_bias=True, name=f'dense_{i}', activation=dense_activation,
                        kernel_regularizer=tf.keras.regularizers.l2(l2=lambda_l2)))
        model.add(Dropout(p_dropout if p_dropout is not None else 0))
        if dense_layer['batch_normalization']:
            model.add(BatchNormalization())

    # Output
    model.add(Dense(n_classes, activation='softmax'))

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)

    # Bind the optimizer and the loss function to the model
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model
