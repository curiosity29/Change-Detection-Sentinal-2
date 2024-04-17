import keras
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet import ResNet50

@keras.saving.register_keras_serializable()
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output



# channels = 6
# IMAGE_SIZE = 256 # Assume square
# NUM_CLASSES = 5

def DeeplabV3Plus(image_size, n_class, channels, **ignore):
    model_input = tf.keras.Input(shape=(image_size, image_size, channels))
    # IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, channels)
    # random crop input image

    # with tensorflow.distribute.Strategy.scope():
    #   input = tensorflow.keras.layers.RandomCrop(height = 256, width = 256)(model_input)
    # print(input.shape)
    # image_size = 256
    # #
    # input_3_channels = layers.Conv2D(256, 1, padding = 'same')(model_input)
    # input_3_channels = tf.nn.relu(input_3_channels)
    # input_3_channels = layers.Conv2D(3, 1, padding = 'same')(input_3_channels)


    resnet50 = tf.keras.applications.ResNet50(input_tensor=model_input,
                                               include_top=False,
                                               weights=None)

    # resnet50 = tf.keras.Model(inputs = resnet_base.layer)
    # resnet50 = ResNet50(
    #     weights=None, include_top=False, input_tensor=input_3_channels
    # )

    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(n_class, kernel_size=(1, 1), padding="same")(x)
    return tf.keras.Model(inputs=model_input, outputs=model_output)