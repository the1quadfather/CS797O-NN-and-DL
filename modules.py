import tensorflow as tf
from tensorflow import keras
from keras import layers

def spp_net(img_in):
    # SPP Net within CSP-SPP
    # Input: 26x26x32; Output: 26x26x128
    # Returns the SPP-Net model ready for use
    if img_in is None:
        inputs = keras.Input(shape=(26, 26, 32))
    else:
        inputs = img_in
    # 3-headed feature extraction max pool layers
    branch1 = layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1))(inputs)
    branch2 = layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1))(inputs)
    branch3 = layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1))(inputs)

    # Check and resize each branch to match expected input size

    if img_in.shape[1:2] != 26:
        img_in = tf.image.resize_with_pad(img_in, 26, 26)
    if branch1.shape[1:2] != 26:
        branch1 = tf.image.resize_with_pad(branch1, 26, 26)
    if branch2.shape[1:2] != 26:
        branch2 = tf.image.resize_with_pad(branch2, 26, 26)
    if branch3.shape[1:2] != 26:
        branch3 = tf.image.resize_with_pad(branch3, 26, 26)

    # Bring the branches together
    output = layers.concatenate([img_in, branch1, branch2, branch3])
    spp_net_model = keras.Model(inputs=inputs, outputs=output, name="SPP-Net")
    spp_net_model.summary()

    return spp_net_model

def lsm():
    # Build LSM model and module
    # Input: 416x416x3 image; Output: 104x104x3 feature map
    # Returns the LSM model ready for use
    # Keras non-sequential model API
    # FYI the model can be shown as a graphic plot using:
    # keras.utils.plot_model(model, "<something.png>")
    inputs = keras.Input(shape=(416, 416, 3))
    conv = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2))
    conv_b1 = conv(inputs)
    # Input layer into first convolutional layer. Now branch:
    # Branch 1
    conv_b1 = layers.Conv2D(16, kernel_size=(1, 1), strides=(1, 1))(conv_b1)
    conv_b1 = layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2))(conv_b1)
    output_b1 = layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1))(conv_b1)

    # Branch 2
    conv_b2 = conv(inputs)
    conv_b2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv_b2)
    output_b2 = layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1))(conv_b2)

    # Concacenation
    output = layers.concatenate([output_b1, output_b2])

    lsm_model = keras.Model(inputs=inputs, outputs=output, name="LSM")
    lsm_model.summary()


    # tf.keras.utils.plot_model(model, "lslnet_trial.png")
    # TODO: Not sure why plot_model doesn't want to work. Try in Colab
    # TODO: Layer output shapes are 1 off in both dimensions; problem?

    return lsm_model

def csp():
    # Build CSP model and module
    # Input: 104x104x64 feature map from LSM
    # Output: Reduced feature maps
    # Returns the CSP model ready for use
    inputs = keras.Input(shape=(104, 104, 64), batch_size=1)

    csp_filters = 32
    # Initial convolutional layer
    conv = layers.Conv2D(csp_filters, kernel_size=(3, 3), strides=(1, 1))
    conv1 = conv(inputs)

    # Additional feature extraction branch(es)
    conv_b1 = layers.Conv2D(csp_filters, kernel_size=(3, 3), strides=(1, 1))(conv1)
    conv_b1_1 = layers.Conv2D(csp_filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(conv_b1)
    output_b1_1 = layers.concatenate([conv_b1, conv_b1_1])
    output_b1 = layers.Conv2D(csp_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")(output_b1_1)
    output_b1 = layers.MaxPooling2D(pool_size=(1, 1), padding="same")(output_b1)
    output_b1 = tf.image.resize_with_pad(output_b1, 102, 102)
    # Final concatenation and feature map output
    output = layers.concatenate([conv1, output_b1])

    csp_model = keras.Model(inputs=inputs, outputs=output, name="CSP")
    csp_model.summary()

    return csp_model

def csp_spp():
    # CSP-SPP submodule within the EFM
    # Input: Image of shape 26x26x256; Output: 13x13x256 feature map
    # Returns the CSP-SPP model ready for use
    inputs = keras.Input(shape=(26, 26, 256))
    # First convolution
    conv = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1))
    conv1 = conv(inputs)

    # Branch B
    strides = (1, 1)
    conv_b = layers.Conv2D(128, kernel_size=(3, 3), strides=strides)(conv1)
    conv_b = layers.Conv2D(64, kernel_size=(1, 1), strides=strides)(conv_b)
    # Continues Branch B into B1 (b is used later for a recurrent connection)
    conv_b1 = layers.Conv2D(64, kernel_size=(3, 3), strides=strides)(conv_b)
    conv_b1 = layers.Conv2D(32, kernel_size=(1, 1), strides=strides)(conv_b1)
    conv_b1_spp = spp_net(conv_b1)
    conv_b1 = conv_b1_spp(conv_b1)
    conv_b = tf.image.resize_with_pad(conv_b1, 26, 26)
    conc1 = layers.concatenate([conv_b, conv_b1])
    conv_B = layers.Conv2D(256, kernel_size=(3, 3), strides=strides)(conc1)

    # Final concatenation
    conc2 = layers.concatenate([conv1, conv_B])

    # Max pooling layer
    endpool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conc2)
    output = layers.Conv2D(256, kernel_size=(1, 1), strides=strides)(endpool)

    csp_spp_model = keras.Model(inputs=inputs, outputs=output, name="CSP-SPP")
    csp_spp_model.summary()

    return csp_spp_model

def efm(input_tensor):
    if input_tensor.shape[1:2] != 104:
        input_tensor = tf.image.resize_with_pad(input_tensor, 104, 104)
    efm_output = csp()(input_tensor)
    if efm_output.shape[1:2] != 104:
        efm_output = tf.image.resize_with_pad(efm_output, 104, 104)
    efm_output = csp()(efm_output)
    if efm_output.shape[1:2] != 104:
        efm_output = tf.image.resize_with_pad(efm_output, 104, 104)
    efm_output_attn = keras.layers.Attention()([efm_output, input_tensor])
    efm_output_attn = tf.reshape(efm_output_attn, (-1, 26, 26, 256))

    efm_output = csp_spp()(efm_output_attn)

    return efm_output

