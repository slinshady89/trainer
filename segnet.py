from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from layers import MaxUnpooling2D, MaxPoolingWithArgmax2D




def segnet(input_shape,
            n_labels,
            filters = 16,
            kernel = 3,
            pool_size = (2, 2),
            output_mode = "softmax"):

    inputs = Input(shape = input_shape)
    activation = 'relu'
    # encoder
    print('\nBuilding UNet with pooling indices\n')
    conv_1 = Layer(activation, 4, filters, inputs, kernel)
    conv_2 = Layer(activation, 4, filters, conv_1, kernel)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Layer(activation, 8, filters, pool_1, kernel)
    conv_4 = Layer(activation, 8, filters, conv_3, kernel)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Layer(activation, 16, filters, pool_2, kernel)
    conv_6 = Layer(activation, 16, filters, conv_5, kernel)
    conv_7 = Layer(activation, 16, filters, conv_6, kernel)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Layer(activation, 32, filters, pool_3, kernel)
    conv_9 = Layer(activation, 32, filters, conv_8, kernel)
    conv_10 = Layer(activation, 32, filters, conv_9, kernel)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Layer(activation, 32, filters, pool_4, kernel)
    conv_12 = Layer(activation, 32, filters, conv_11, kernel)
    conv_13 = Layer(activation, 32, filters, conv_12, kernel)


    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build encoder done..")


    # decoder
    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Layer(activation, 32, filters, unpool_1, kernel)
    conv_15 = Layer(activation, 32, filters, conv_14, kernel)
    conv_16 = Layer(activation, 32, filters, conv_15, kernel)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Layer(activation, 32, filters, unpool_2, kernel)
    conv_18 = Layer(activation, 32, filters, conv_17, kernel)
    conv_19 = Layer(activation, 16, filters, conv_18, kernel)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Layer(activation, 16, filters, unpool_3, kernel)
    conv_21 = Layer(activation, 16, filters, conv_20, kernel)
    conv_22 = Layer(activation, 8, filters, conv_21, kernel)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Layer(activation, 8, filters, unpool_4, kernel)
    conv_24 = Layer(activation, 4, filters, conv_23, kernel)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Layer(activation, 4, filters, unpool_5, kernel)

    conv_26 = Convolution2D(n_labels, (1, 1), padding = "valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    segunet = Model(inputs = inputs, outputs = outputs, name = "ContiPathNet")

    return segunet


def Layer(activation, filter_factor, filters, inputs, kernel):
    conv_1 = Convolution2D(filters * filter_factor, (kernel, kernel), padding = "same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation(activation)(conv_1)
    return conv_1
