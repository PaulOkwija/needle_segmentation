#AGATA's APPROACH
import tensorflow as tf
from keras.models import Model
from tensorflow.keras import layers, regularizers



def model(input_shape):
    inputs = layers.Input(input_shape, dtype=tf.float32)
    act = 'relu'
    kinit = tf.keras.initializers.GlorotUniform()
    #inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    conv_16 = layers.Conv2D(16, (11,11) , padding='same')(inputs)
    conv_16 = layers.BatchNormalization(axis=3)(conv_16)
    relu_16 = layers.Activation(act)(conv_16)
    pool_16 = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(relu_16)

    
    conv_32 = layers.Conv2D(32, (9,9), padding='same')(pool_16)
    conv_32 = layers.BatchNormalization(axis=3)(conv_32)
    relu_32 = layers.Activation(act)(conv_32)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)

    
    conv_64 = layers.Conv2D(64, (7,7), padding='same')(relu_32)
    conv_64 = layers.BatchNormalization(axis=3)(conv_64)
    relu_64 = layers.Activation(act)(conv_64)
    pool_64 = layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(conv_64)

    
    up_64 = layers.Conv2DTranspose(64, (7, 7), strides=(1,1), padding='same')(pool_64)
    up_64 = layers.BatchNormalization(axis=3)(up_64)
    relu_up_64 = layers.Activation(act)(up_64)
    # up_64 = layers.MaxPooling2D(pool_size=(2,2))(up_64)

    
    up_128 = layers.Conv2DTranspose(32, (9, 9), strides=(2,2), padding='same')(relu_up_64)
    up_128 = layers.BatchNormalization(axis=3)(up_128)
    relu_up_128 = layers.Activation(act)(up_128)
    # up_128 = layers.MaxPooling2D(pool_size=(2,2))(up_128)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(1, kernel_size=(1,1))(relu_up_128)
    sigmoid_final = layers.Activation('sigmoid')(conv_final)
    #Change to softmax for multichannel

    # Model integration
    model = Model(inputs, sigmoid_final, name="Unet_up_2")
    # model.compile(optimizer='adam', loss=[jaccard_coef_loss], metrics=[jaccard_coef])
    # model.summary()
    return model