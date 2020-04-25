import tensorflow as tf
import numpy as np 

#module for the single BlazeFace block.
def singleBlazeBlock(x, filters=24, kernel_size=5, strides=1, channel_padding =  False,padding='same'):

    # depth-wise separable convolution
    x_0 = tf.keras.layers.SeparableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)
    x_1 = tf.keras.layers.BatchNormalization()(x_0)
   
    # Residual connection
    if strides == 2:
        #if strides == 2 then we need to do max pooling before adding the input and output tensors.
        x_ = tf.keras.layers.MaxPooling2D()(x)
        #if the input and output channels are different then channel padding is applied.
        if channel_padding:
            x_ = tf.keras.layers.concatenate([x_, tf.zeros_like(x_)], axis=-1)

        out = tf.keras.layers.Add()([x_1, x_])
        return tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.Add()([x_1, x])
    return tf.keras.layers.Activation("relu")(out)


def doubleBlazeBlock(x, filters_1=24, filters_2=96,
                     kernel_size=5, strides=1, channel_padding = False,padding='same'):

    # depth-wise separable convolution, project
    x_0 = tf.keras.layers.SeparableConv2D(
        filters=filters_1,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    x_1 = tf.keras.layers.BatchNormalization()(x_0)

    x_2 = tf.keras.layers.Activation("relu")(x_1)

    # depth-wise separable convolution, expand
    x_3 = tf.keras.layers.SeparableConv2D(
        filters=filters_2,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        use_bias=False)(x_2)

    x_4 = tf.keras.layers.BatchNormalization()(x_3)

    # Residual connection
    if strides == 2:
        #if strides == 2 then we need to do max pooling before adding the input and output tensors.
        x_ = tf.keras.layers.MaxPooling2D()(x)
        #if the input and output channels are different then channel padding is applied.
        if channel_padding:
            x_ = tf.keras.layers.concatenate([x_, tf.zeros_like(x_)], axis=-1)

        out = tf.keras.layers.Add()([x_4, x_])
        return tf.keras.layers.Activation("relu")(out)
    
    out = tf.keras.layers.Add()([x_4, x])
    return tf.keras.layers.Activation("relu")(out)


def network(input_shape):

    inputs = tf.keras.layers.Input(shape=input_shape)

    #The following architecture is derived from the paper.
    x_0 = tf.keras.layers.Conv2D(
        filters=24, kernel_size=5, strides=2, padding='same')(inputs)
    x_0 = tf.keras.layers.BatchNormalization()(x_0)
    x_0 = tf.keras.layers.Activation("relu")(x_0)

    # single BlazeBlock phase
    x_1 = singleBlazeBlock(x_0)
    x_2 = singleBlazeBlock(x_1)
    x_3 = singleBlazeBlock(x_2, strides=2, filters=48, channel_padding = True)
    x_4 = singleBlazeBlock(x_3, filters=48)
    x_5 = singleBlazeBlock(x_4, filters=48)

    # double BlazeBlock phase
    x_6 = doubleBlazeBlock(x_5, strides=2, channel_padding = True)
    x_7 = doubleBlazeBlock(x_6)
    x_8 = doubleBlazeBlock(x_7)
    x_9 = doubleBlazeBlock(x_8, strides=2)
    x_10 = doubleBlazeBlock(x_9)
    x_11 = doubleBlazeBlock(x_10)

    model = tf.keras.models.Model(inputs=inputs, outputs=[x_8, x_11])
    #inputs to the model : [batch_size, 128,128,3]
    #outputs : feature maps of sizes (batch_size,16,16,96) and (batch_size,8,8,96)
    return model

#testing the BlazeFace architecture.
x = np.float32(np.random.random((3, 128, 128, 3)))
blazeface_extractor = network((128, 128, 3))
feature = blazeface_extractor(x)
assert feature[0].shape == (3, 16, 16, 96) or feature[1].shape == (3, 8, 8, 96)
