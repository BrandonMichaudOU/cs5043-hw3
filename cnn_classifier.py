import tensorflow as tf
from keras.layers import (InputLayer, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D,
                          GlobalMaxPooling2D, SpatialDropout2D)
from tensorflow.keras.models import Sequential


def create_cnn_classifier_network(image_size, nchannels, conv_layers=None, dense_layers=None, p_dropout=None,
                                  p_spatial_dropout=None, lambda_l2=None, lrate=0.001, n_classes=4,
                                  loss='sparse_categorical_crossentropy', metrics=None, padding='valid',
                                  conv_activation='elu', dense_activation='elu'):
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
        model.add(SpatialDropout2D(p_spatial_dropout))
        if conv_layer['batch_normalization']:
            model.add(BatchNormalization())

    # Global max pooling
    model.add(GlobalMaxPooling2D())

    # Dense modules
    for i, dense_layer in enumerate(dense_layers):
        model.add(Dense(dense_layer['units'], use_bias=True, name=f'dense_{i}', activation=dense_activation,
                        kernel_regularizer=tf.keras.regularizers.l2(l2=lambda_l2)))
        model.add(Dropout(p_dropout))
        if dense_layer['batch_normalization']:
            model.add(BatchNormalization())

    # Output
    model.add(Dense(n_classes, activation='softmax'))

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)

    # Bind the optimizer and the loss function to the model
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    # Generate an ASCII representation of the architecture
    print(model.summary())
    return model
