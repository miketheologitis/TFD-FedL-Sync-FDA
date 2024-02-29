import tensorflow as tf
from tensorflow.keras import layers, models


def sequential_lenet5(cnn_input_reshape=(28, 28, 1), num_classes=10):
    """
    Args:
    - cnn_input_reshape (tuple): The shape to which the input should be reshaped (e.g., (28, 28, 1)).
    - num_classes (int): Number of output classes.

    Returns:
    - tf.keras.models.Sequential: A LeNet-5 model using the Sequential API.

    Example for MNIST:
      lenet5 = sequential_lenet5((28, 28, 1), 10)
      lenet5.compile(...)
      lenet5.fit(...)
    """
    return tf.keras.models.Sequential([
        # Reshape layer
        tf.keras.layers.Reshape(cnn_input_reshape, input_shape=(28, 28)),  # Example input shape, change as needed
        # Layer 1 Conv2D
        tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same'),
        # Layer 2 Pooling Layer
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        # Layer 3 Conv2D
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
        # Layer 4 Pooling Layer
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        # Flatten
        tf.keras.layers.Flatten(),
        # Layer 5 Dense
        tf.keras.layers.Dense(units=120, activation='tanh'),
        # Layer 6 Dense
        tf.keras.layers.Dense(units=84, activation='tanh'),
        # Layer 7 Dense
        tf.keras.layers.Dense(units=num_classes, activation='softmax')  # Example num_classes=10, change as needed
    ])


def sequential_advanced_cnn(cnn_input_reshape=(28, 28, 1), num_classes=10):
    """
    Create the AdvancedCNN using the Sequential API.

    Args:
    - cnn_input_reshape (tuple): The shape to which the input should be reshaped (e.g., (28, 28, 1)).
    - num_classes (int): Number of output classes.

    Returns:
    - tf.keras.models.Sequential: An AdvancedCNN model using the Sequential API.

    Example for MNIST:
      advanced_cnn = sequential_advanced_cnn((28, 28, 1), 10)
      advanced_cnn.compile(...)
      advanced_cnn.fit(...)
    """
    return tf.keras.models.Sequential([
        # Reshape layer
        tf.keras.layers.Reshape(cnn_input_reshape, input_shape=(28, 28)),  # Example input shape, change as needed
        # First Convolutional Block
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Second Convolutional Block
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Third Convolutional Block
        tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Fully Connected Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


""" 
Implementation from https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py

DenseNet, not pre-trained, specifically for the CIFAR-10 datasets. 

Note:
    - Beforehand preprocessing on input is assumed using `tensorflow.keras.applications.densenet.preprocess_input`.
    - We assume NHWC input data format (which we then handle internally).

Deviations from original keras implementation:
    1) We add dropout layers with rate=0.2 as suggested by Huang et. al, 2016 for training on CIFAR-10
    2) We adopt `he normal` weight-initialization He et al., 2015 as suggested by Huang et. al., 2016
    3) We adopt NCHW data format (channels first). The input is expected to be in NHWC format which we then transform
        to NCHW format. This is due to layout optimization error/explanation of in
        https://github.com/tensorflow/tensorflow/issues/34499#issuecomment-652316457
    4) We do not put `weight_decay=1e-4` in the `optimizers.SGD` but rather equivalently replicate it by putting
        `regularizers.L2(1e-4)` on the weights (kernel) of the `Conv2D` and `Dense` layers. This is because in TF 2.7
        `weight_decay` is not available in the optimizer.
"""


def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    bn_axis = 1  # For NCHW format : (batch_size, channels, height, width)

    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=name + '_bn'
    )(x)

    x = layers.Activation('relu', name=name + '_relu')(x)

    x = layers.Conv2D(
        int(x.shape[bn_axis] * reduction),
        1,
        kernel_initializer='he_normal',
        use_bias=False,
        name=name + '_conv',
        data_format='channels_first',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x)

    x = layers.AveragePooling2D(
        2,
        strides=2,
        name=name + '_pool',
        data_format='channels_first'
    )(x)

    return x


def conv_block(x, growth_rate, name):
    bn_axis = 1  # For NCHW format : (batch_size, channels, height, width)

    x1 = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=name + '_0_bn'
    )(x)

    x1 = layers.Activation(
        'relu',
        name=name + '_0_relu'
    )(x1)

    x1 = layers.Conv2D(
        4 * growth_rate,
        1,
        use_bias=False,
        kernel_initializer='he_normal',
        name=name + '_1_conv',
        data_format='channels_first',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x1)

    x1 = layers.Dropout(
        0.2,
        name=name + '_1_dropout'
    )(x1)  # Add dropout 0.2 after convolution as Huang et. al suggest for Cifar-10

    x1 = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=name + '_1_bn'
    )(x1)

    x1 = layers.Activation(
        'relu',
        name=name + '_1_relu'
    )(x1)

    x1 = layers.Conv2D(
        growth_rate,
        3,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        name=name + '_2_conv',
        data_format='channels_first',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x1)

    x1 = layers.Dropout(
        0.2,
        name=name + '_2_dropout'
    )(x1)  # Add dropout 0.2 after convolution as Huang et. al suggest for Cifar-10

    x = layers.Concatenate(
        axis=bn_axis,
        name=name + '_concat'
    )([x, x1])

    return x


def dense_net_fn(blocks, input_shape, classes):
    # Determine proper input shape
    img_input = layers.Input(shape=input_shape)

    bn_axis = 1  # For NCHW format : (batch_size, channels, height, width)

    x_nchw = tf.transpose(img_input, [0, 3, 1, 2])  # Transform to NCHW format

    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)),
        data_format='channels_first'
    )(x_nchw)

    x = layers.Conv2D(
        64,
        7,
        strides=2,
        use_bias=False,
        kernel_initializer='he_normal',
        name='conv1/conv',
        data_format='channels_first',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x)

    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name='conv1/bn'
    )(x)

    x = layers.Activation(
        'relu',
        name='conv1/relu'
    )(x)

    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)),
        data_format='channels_first'
    )(x)

    x = layers.MaxPooling2D(
        3,
        strides=2,
        name='pool1',
        data_format='channels_first'
    )(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name='bn'
    )(x)

    x = layers.Activation(
        'relu',
        name='relu'
    )(x)

    x = layers.GlobalAveragePooling2D(
        name='avg_pool',
        data_format='channels_first'
    )(x)

    x = layers.Dense(
        classes,
        kernel_initializer='he_normal',
        activation='softmax',
        name='fc10',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x)

    inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = models.Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = models.Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = models.Model(inputs, x, name='densenet201')
    else:
        model = models.Model(inputs, x, name='densenet')

    return model


def get_densenet(name, input_shape=(32, 32, 3), classes=10):
    model = None

    if name == 'DenseNet121':
        model = dense_net_fn([6, 12, 24, 16], input_shape, classes)
    if name == 'DenseNet169':
        model = dense_net_fn([6, 12, 32, 32], input_shape, classes)
    if name == 'DenseNet201':
        model = dense_net_fn([6, 12, 48, 32], input_shape, classes)
    return model


def build_and_compile_advanced_cnn_for_mnist():
    adv = sequential_advanced_cnn()

    adv.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        ),  # we have softmax
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    adv.build((None, 28, 28))

    return adv


def build_and_compile_lenet5_for_mnist():
    lenet = sequential_lenet5()

    lenet.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        ),  # we have softmax
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    lenet.build((None, 28, 28))

    return lenet


def build_and_compile_densenet_for_cifar10(name):
    densenet = get_densenet(name)

    densenet.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        ),  # we have softmax
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    densenet.build((None, 32, 32, 3))

    return densenet



