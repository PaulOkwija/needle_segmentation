#VERSION _ 1
import tensorflow as tf 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
kinit = 'he_normal'


def expend_as(tensor, rep,name):
	my_repeat = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},  name='psi_up'+name)(tensor)
	return my_repeat


def AttnGatingBlock(x, g, inter_shape, name):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x 
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
    
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer=kinit,padding='same', name='xl'+name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = layers.Conv2D(inter_shape, (1, 1), kernel_initializer=kinit,padding='same')(g)
    upsample_g = layers.Conv2DTranspose(inter_shape, (2, 2),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same', name='g_up'+name)(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), kernel_initializer=kinit,padding='same', name='psi'+name)(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3],  name)
    y = layers.multiply([upsample_psi, x], name='q_attn'+name)

    result = layers.Conv2D(shape_x[3], (1, 1),kernel_initializer=kinit, padding='same',name='q_attn_conv'+name)(y)
    result_bn = layers.BatchNormalization(axis=3,name='q_attn_bn'+name)(result)
    return result_bn

def UnetConv2D(input, outdim, is_batchnorm, name):
    x = layers.Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_1')(input)
    if is_batchnorm:
      x =layers.BatchNormalization(axis=3,name=name + '_1_bn')(x)
    x = layers.Activation('relu',name=name + '_1_act')(x)
  
    x = layers.Dropout(0.1)(x)
  
    
    x = layers.Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_2')(x)
    if is_batchnorm:
      x = layers.BatchNormalization(axis=3,name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_act')(x)
    return x
	

def UnetGatingSignal(input, is_batchnorm, name):
    ''' this is simply 1x1 convolution, bn, activation '''
    shape = K.int_shape(input)
    x = layers.Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same",  kernel_initializer=kinit, name=name + '_conv')(input)
    if is_batchnorm:
        x = layers.BatchNormalization(name=name + '_bn')(x)
    x = layers.Activation('relu', name = name + '_act')(x)
    return x

# plain old attention gates in u-net, NO multi-input, NO deep supervision
def model(opt,input_size):   
    inputs = layers.Input(shape=input_size)
    conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')
    #conv3 = layers.Dropout(0.2,name='drop_conv3')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 256, is_batchnorm=True, name='conv4')
    #conv4 = layers.Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    center = UnetConv2D(pool4, 512, is_batchnorm=True, name='center')
    
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 256, '_1')
    up1 = layers.concatenate([layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')
    up_conv1 = UnetConv2D(up1, 256, is_batchnorm=True, name='up_conv1')
    
    
    g2 = UnetGatingSignal(up_conv1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 128, '_2')
    up2 = layers.concatenate([layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up_conv1), attn2], name='up2')
    up_conv2 = UnetConv2D(up2, 128, is_batchnorm=True, name='up_conv2')

    g3 = UnetGatingSignal(up_conv2, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 64, '_3')
    up3 = layers.concatenate([layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up_conv2), attn3], name='up3')
    up_conv3 = UnetConv2D(up3, 64, is_batchnorm=True, name='up_conv3')

    g4 = UnetGatingSignal(up_conv3, is_batchnorm=True, name='g4')
    attn4 = AttnGatingBlock(conv1, g4, 32, '_4')
    up4 = layers.concatenate([layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up_conv3), attn4], name='up4')
    up_conv4 = UnetConv2D(up4, 32, is_batchnorm=True, name='up_conv4')


    out = layers.Conv2D(1, (1, 1), activation='sigmoid',  kernel_initializer=kinit, name='final')(up_conv4)
    
    model = Model(inputs=[inputs], outputs=[out])
    # model.compile(optimizer=opt, loss=lossfxn, metrics=[losses.dsc,losses.tp,losses.tn])
    return model