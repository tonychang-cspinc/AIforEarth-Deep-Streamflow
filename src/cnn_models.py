from keras import models, layers
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Dropout, \
        Flatten, Dense, Concatenate, Input
from keras.models import Model, load_model
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU
from keras.applications import densenet
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from src.dense_net import dense_stack
from keras import backend as K

K.set_image_data_format('channels_last')

def rmse(y_true, y_pred):
    return(K.sqrt(K.mean(K.square(y_pred - y_true))))

def det_coeff(y_true, y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return(K.ones_like(v) - (u / v))

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return(1 - (SS_res/(SS_tot + K.epsilon())))

def conv_block(x,f):
    x = Conv2D(f, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)   
    return(x)

def basic_cnn(input_shape, output_shape, final_activation='softmax'):
    input_layer = Input(shape=input_shape, name='input_layer')
    x = conv_block(input_layer,16)
    x = BatchNormalization()(x)
    x = Dense(10)(x)
    #x = Flatten()(x)
    if final_activation is not None:
        output_layer = Dense(output_shape, activation=final_activation, name='output_layer')(x)
    else:
        output_layer = Dense(output_shape, name='output_layer')(x)
    return(models.Model(inputs=input_layer, outputs=output_layer))


def simple_cnn(input_shape, output_shape, final_activation='softmax'):
    input_layer = Input(shape=input_shape, name='input_layer')
    x = conv_block(input_layer,16)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = conv_block(x,32)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = BatchNormalization()(x)
    x = Dense(20)(x)
    x = Flatten()(x)
    if final_activation is not None:
        output_layer = Dense(output_shape, activation=final_activation, name='output_layer')(x)
    else:
        output_layer = Dense(output_shape, name='output_layer')(x)
    return(models.Model(inputs=input_layer, outputs=output_layer))

def dense_cnn(input_shape, n_dblocks, output_shape, final_activation='softmax'):
    input_layer = Input(shape=input_shape, name='input_layer')
    dense = dense_stack(input_layer, n_final_filter=64, \
            dropout=0.1, n_dblocks=n_dblocks)
    fc1 = Dense(256, name='fc1')(dense)
    fc2 = Dense(128, name='fc2')(fc1)
    x = Dense(64)(fc2)
    x = Flatten()(x)
    if final_activation is not None:
        x = Activation(final_activation)(x)   
    output_layer = Dense(output_shape, name='output_layer')(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return(model)

def dense_cnn_multitask(input_shape, n_dblocks, output_shape):
    input_layer = Input(shape=input_shape, name='input_layer')
    dense = dense_stack(input_layer, n_final_filter=64, \
            dropout=0.1, n_dblocks=n_dblocks)
    fc1 = Dense(256, name='fc1')(dense)
    fc2 = Dense(128, name='fc2')(fc1)
    fc_class = Dense(128)(fc2)
    fc_flow = Dense(128)(fc2)
    fc_class = Dense(64)(fc_class)
    fc_flow = Dense(64)(fc_flow)
    fcc = Dense(output_shape[0])(fc_class)
    fcf = Dense(output_shape[1])(fc_flow)
    fcc = Activation('softmax', \
                        name='fc_class_layer')(fcc)
    fcf = Activation('relu', \
                        name='fc_flow_layer')(fcf)

    output_layer = [fcc, fcf]
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return(model)

def transfer_densenet_model(input_shape, n_classes):
    base_model = densenet.DenseNet121(input_shape=input_shape,
    weights='imagenet',
    include_top=False,
    pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = Dense(1000,\
            kernel_regularizer=regularizers.l1_l2(0.01),\
            activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500,\
            kernel_regularizer=regularizers.l1_l2(0.01),\
            activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model
