from keras import layers
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, Flatten, UpSampling2D, AveragePooling2D, BatchNormalization, Dense
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import RandomNormal, constant
#from keras.backend import tf as ktf
import tensorflow as tf 

from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.applications.densenet import DenseNet121
from keras.applications.resnet import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2

import re
import numpy as np
from math import ceil

def relu(x): return Activation('relu')(x)

def fc(x, nf, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    x = Dense(nf, name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=RandomNormal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=RandomNormal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block_keras(x, weight_decay, model_name='vgg19', layer_name='block4_conv4', weights='imagenet', upsampling=False):
    if x.get_shape()[3] != 3:
        model = VGG19(include_top=False, weights=weights, input_shape=(x.get_shape()[1], x.get_shape()[2], 3))
        first_layer = model.layers[1]
        
        new_first_layer = Conv2D(filters=first_layer.filters, kernel_size=first_layer.kernel_size, strides=first_layer.strides, padding=first_layer.padding, data_format=first_layer.data_format, dilation_rate=first_layer.dilation_rate, activation=first_layer.activation, use_bias=first_layer.use_bias, kernel_regularizer=first_layer.kernel_regularizer, bias_regularizer=first_layer.bias_regularizer, activity_regularizer=first_layer.activity_regularizer, kernel_constraint=first_layer.kernel_constraint, bias_constraint=first_layer.bias_constraint)
        y = new_first_layer(x)

        w = first_layer.get_weights()[0]
        w_new = np.zeros((w.shape[0], w.shape[1], x.get_shape()[3], w.shape[3]))
        if w.shape[2] < x.get_shape()[3]:
            w_new[:,:,0:w.shape[2],:] = w
            pos = 0
            for i in range(w.shape[2],x.get_shape()[3]):
                w_new[:,:,i,:] = w[:,:,pos,:]
                pos = (pos+1) % w.shape[2]
        else:
            w_new[:,:,:,:] = w[:,:,:x.get_shape()[3],:]

        b_new = first_layer.get_weights()[1]
        new_first_layer.set_weights([w_new, b_new])
        for l in range(2,len(model.layers)):
            y = model.layers[l](y)
            if  model.layers[l].name == layer_name:
                break
        x = y
    else:
        if model_name == 'vgg19':
            # block4_conv4 (1024,1024,3)->(128,128,512)
            # block5_conv4 (1024,1024,3)->(64,64,512)
            # block5_pool (1024,1024,3)->(32,32,512)
            model = VGG19(include_top=False, weights=weights, input_tensor=x)
        elif model_name == 'inceptionresnet':
            # activation_75 (1024,1024,3)->(125,125,256)
            # activation_162 (1024,1024,3)->(62,62,288)
            # conv_7b_ac (1024,1024,3)->(30,30,1536)
            model = InceptionResNetV2(include_top=False, weights=weights, input_tensor=x)
        elif model_name == 'densenet121':
            # pool3_conv (1024,1024,3)->(128,128,256)
            # pool4_conv (1024,1024,3)->(64,64,512)
            # relu (1024,1024,3)->(32,32,1024)
            model = DenseNet121(include_top=False, weights=weights, input_tensor=x)
        elif model_name == 'resnet50':
            # conv3_block4_out (1024,1024,3)->(128,128,512)
            # conv4_block6_out (1024,1024,3)->(64,64,1024)
            # conv5_block3_out (1024,1024,3)->(32,32,2048)
            model = ResNet50(include_top=False, weights=weights, input_tensor=x)
        x = model.get_layer(layer_name).output
    
    if upsampling:
        x = UpSampling2D()(x)
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)

    kernel_reg = l2(weight_decay) if weight_decay else None
    bias_reg = l2(0) if weight_decay else None
    for layer in model.layers:
            if type(layer) is Conv2D:
                layer.kernel_regularizer = kernel_reg
                layer.bias_regularizer = bias_reg

    return x

def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))
    return x

def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    return x


def get_training_model(img_w, img_h, img_d=3, n_frames=2, model_name='vgg19', layer_name='block4_conv4', weights='imagenet', weight_decay = 5e-4, stages = 6, upsampling=False, psp=False, np_branch1 = 2, np_branch2 = 1, np_branch3 = 1):

    img_input_shape = (img_w, img_h, img_d*n_frames)
    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    inputs.append(img_input)
    img_normalized = img_input # will be done on augmentation stage

    # VGG
    #stage0_out = vgg_block(img_normalized, weight_decay)
    stage0_out = vgg_block_keras(img_normalized, weight_decay, model_name, layer_name, weights, upsampling)

    if psp:
        output_shape = (stage0_out.get_shape()[1].value, stage0_out.get_shape()[2].value)
        stage0_out = build_pyramid_pooling_module(stage0_out, img_input_shape, output_shape)

    
    
    new_x = []

    # stage 1 - branch 1 (vetores)
    if np_branch1 > 0:
        stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
        outputs.append(stage1_branch1_out)
        new_x.append(stage1_branch1_out)

    if np_branch2 > 0:
        # stage 1 - branch 2 (confidence maps)
        stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
        outputs.append(stage1_branch2_out)
        new_x.append(stage1_branch2_out)

    # stage 1 - branch 3 (lines)
    if np_branch3 > 0:
        stage1_branch3_out = stage1_block(stage0_out, np_branch3, 3, weight_decay)
        outputs.append(stage1_branch3_out)
        new_x.append(stage1_branch3_out)

    new_x.append(stage0_out)

    x = Concatenate()(new_x)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        new_x = []
         #stage SN - branch 1 (PAF)
        if np_branch1 > 0:
            stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
            outputs.append(stageT_branch1_out)
            new_x.append(stageT_branch1_out)

        if np_branch2 > 0:
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
            outputs.append(stageT_branch2_out)
            new_x.append(stageT_branch2_out)

        if np_branch3 > 0:
            stageT_branch3_out = stageT_block(x, np_branch3, sn, 3, weight_decay)
            outputs.append(stageT_branch3_out)
            new_x.append(stageT_branch3_out)

        new_x.append(stage0_out)

        if sn < stages:
            x = Concatenate()(new_x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, stage0_out

def get_adj_model(n_sampling, feat_dimension, filters):
    input_shape = (n_sampling, feat_dimension)

    inputs = Input(shape=input_shape)
    x = inputs

    for f in filters:
        x = Conv1D(f, 3, padding="same", activation="relu")(x)
        #x = MaxPooling1D(pool_size=2, strides=2)(x)

    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)

    return Model(inputs=inputs, outputs=x)
