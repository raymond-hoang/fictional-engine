#%%
import os
import sys
import time
import re
from sklearn.metrics import r2_score
#import matplotlib.pyplot as plt
#import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
#import holoviews as hv
#hv.extension('bokeh')
#import hvplot.pandas
#import panel as pn
import annutils
from hyperparams import model_str_def, full_model_str_def, input_prepro_option, group_stations,\
                        exclude_stations, train_frac, window_size, epochs,\
                        train_models, save_results, percentiles, initial_lr,\
                        ndays, nwindows

"""## Read data"""
cwd = os.getcwd()
sys.path.append(cwd)
gdrive_root_path = cwd

data_file = "dsm2_ann_inputs_20220204.xlsx"
data_path = os.path.join(cwd,data_file)

num_sheets = {"dsm2_ann_BaselineData_20220120.xlsx":8,
              "dsm2_ann_inputs_20220204.xlsx":9,
              "dsm2_ann_inputs_20220215.xlsx":10,
              "dsm2_ann_observed_15min.xlsx":9}
num_dataset = {"dsm2_ann_inputs_20220204.xlsx":'simulated',
              "dsm2_ann_inputs_20220215.xlsx":'observed',
              "dsm2_ann_observed_15min.xlsx":'observed_15min'}

NFEATURES = (num_sheets[data_file]-1) # * (ndays + nwindows)

dflist = [pd.read_excel(data_path,i,index_col=0,parse_dates=True) for i in range(num_sheets[data_file])]
df_inpout = pd.concat(dflist[0:(num_sheets[data_file])],axis=1).dropna(axis=0)
dfinps = df_inpout.loc[:,~df_inpout.columns.isin(dflist[num_sheets[data_file]-1].columns)]
dfouts = df_inpout.loc[:,df_inpout.columns.isin(dflist[num_sheets[data_file]-1].columns)]

print(dfinps)
print(dfouts)

#%%
# read station names
output_locations = list(dfouts.columns[~dfouts.columns.str.contains('_dup')])
print(output_locations)

"""Define station groups"""
station_without_groups = {'all':output_locations}
station_with_groups = {'G1':['SSS','RSAC101','RSMKL008'],
                  'G2':['Old_River_Hwy_4','Middle_River_Intake','CCWD_Rock','SLTRM004','RSAN032','RSAN037','SLDUT007','ROLD024','RSAN058','RSAN072','OLD_MID','ROLD059','CHDMC006','CHSWP003','CHVCT000'],
                  'G3':['SLCBN002','SLSUS012','SLMZU011','SLMZU025','RSAC064','RSAC075','RSAC081','RSAN007','RSAC092','RSAN018','Martinez']}
final_groups = {False: station_without_groups, 
                True: station_with_groups}

# create folders to save results
result_folders = ['models','results','images']
for result_folder in result_folders:
    if not os.path.exists(os.path.join(gdrive_root_path, result_folder)):
        os.makedirs(os.path.join(gdrive_root_path, result_folder))

"""## Tensorboard Setup
A log directory to keep the training logs

Tensorboard starts a separate process and is best started from the command line. Open a command window and activate this environment (i.e. keras) and goto the current directory. Then type in
```
tensorboard --logdir=./tf_training_logs/ --port=6006
```
"""

# %load_ext tensorboard
# %tensorboard --logdir=./tf_training_logs/ --port=6006
root_logdir = os.path.join(os.curdir, "tf_training_logs")
tensorboard_cb = keras.callbacks.TensorBoard(root_logdir)## Tensorflow Board Setup

"""## Model definitions

### ResNet Block Building Function
"""

parameters = {
    "kernel_initializer": "he_normal"
}

def basic_1d(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
    force_identity_shortcut=False
):
    """
    A one-dimensional basic block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2


    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.ZeroPadding1D(padding=1,name="padding{}{}_branch2a".format(stage_char, block_char))(x)
        y = keras.layers.Conv1D(filters,kernel_size,strides=stride,use_bias=False,
                                name="res{}{}_branch2a".format(stage_char, block_char),
                                **parameters)(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding1D(padding=1,name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv1D(filters,kernel_size,use_bias=False,
                                name="res{}{}_branch2b".format(stage_char, block_char),
                                **parameters)(y)
        y = keras.layers.BatchNormalization()(y)

        if block != 0 or force_identity_shortcut:
            shortcut = x
        else:
            shortcut = keras.layers.Conv1D(filters,1,strides=stride,use_bias=False,
                                           name="res{}{}_branch1".format(stage_char, block_char),
                                           **parameters)(x)
            shortcut = keras.layers.BatchNormalization()(shortcut)

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        
        y = keras.layers.Activation("relu",name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f

def bottleneck_1d(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
):
    """
    A one-dimensional bottleneck block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    """
    if stride is None:
        stride = 1 if block != 0 or stage == 0 else 2

    # axis = -1 if keras.backend.image_data_format() == "channels_last" else 1


    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.Conv1D(filters,1,strides=stride,use_bias=False,
                                name="res{}{}_branch2a".format(stage_char, block_char),
                                **parameters)(x)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu",name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding1D(padding=1,name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv1D(filters,kernel_size,use_bias=False,
                                name="res{}{}_branch2b".format(stage_char, block_char),
                                **parameters)(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu",name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = keras.layers.Conv1D(filters * 4, 1, use_bias=False,
                                name="res{}{}_branch2c".format(stage_char, block_char),
                                **parameters)(y)
        y = keras.layers.BatchNormalization()(y)

        if block == 0:
            shortcut = keras.layers.Conv1D(filters * 4, 1, strides=stride, use_bias=False,
                                           name="res{}{}_branch1".format(stage_char, block_char),
                                           **parameters)(x)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = keras.layers.Activation("relu",name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f

"""# Custom loss function"""
def mse_loss_masked(y_true, y_pred):
    squared_diff = tf.reduce_sum(tf.math.squared_difference(y_pred[y_true>0],y_true[y_true>0]))
    return squared_diff/(tf.reduce_sum(tf.cast(y_true>0, tf.float32))+0.01)

# Define Sequential model

"""### Layer builders"""
def build_layer_from_string_def(s='i120',width_multiplier=1,
                                block=0,
                                force_identity_shortcut=False,
                                return_sequences_rnn=True):
    if s[0:4] == 'lstm':
        return layers.LSTM(units = int(s[4:])*width_multiplier, return_sequences=return_sequences_rnn, activation='sigmoid')
    elif s[0:3] == 'res':
        fields = s[3:].split('x')
        return basic_1d(filters=int(fields[0]),
                        stage=int(fields[3]),
                        block=block,
                        kernel_size=int(fields[1]),
                        stride=int(fields[2]),
                        force_identity_shortcut=force_identity_shortcut)
    elif s[0:3] == 'c1d':
        fields = s[3:].split('x')
        return keras.layers.Conv1D(filters=int(fields[0]), kernel_size=int(fields[1]), strides=int(fields[2]),
                                   padding='causal', activation='linear')
    elif s[0:2] == 'td':
        return keras.layers.TimeDistributed(keras.layers.Dense(int(s[2:]), activation='elu'))
    elif s[0:2] == 'dr':
        return keras.layers.Dropout(float(s[2:]))
    # elif s[0] == 'i':
    #     return keras.layers.InputLayer(input_shape=[int(s[1:]), NFEATURES])
    elif s[0] == 'f':
        return keras.layers.Flatten()
    elif s[0] == 'g':
        return keras.layers.GRU(int(s[1:])*width_multiplier, return_sequences=True, activation='relu')
    elif s[0] == 'd':
        return keras.layers.Dense(int(s[1:])*width_multiplier, activation='elu')
    elif s[0] == 'o':
        return keras.layers.Dense(int(s[1:])*width_multiplier, activation='linear')
    else:
        raise Exception('Unknown layer def: %s' % s)

"""### Model builders"""
def build_model_from_string_def(strdef='i120_f_d4_d2_d1',width_multiplier=1):
    layer_strings = strdef.split('_')
    inputs = keras.layers.Input(shape=[int(layer_strings[0][1:]) * NFEATURES])
    x = None
    prev_conv_output_num_of_channels = None
    return_sequences_rnn = None
    for block,f in enumerate(layer_strings[1:-1]):
        if x is None:
            if f.startswith(('lstm','g')):         
                # these layers require 2D inputs and permutation
                x = layers.Reshape((ndays+nwindows,NFEATURES))(inputs)
                prev_conv_output_num_of_channels = NFEATURES
                x = layers.Permute((2,1))(x)
                return_sequences_rnn = layer_strings[block+2].startswith(('lstm','g','res','c1d'))
            elif f.startswith(('res','c1d')):
                # these layers require 2D inputs
                x = layers.Reshape((ndays+nwindows,NFEATURES))(inputs)
                prev_conv_output_num_of_channels = NFEATURES
            else:
                x = inputs


        x = build_layer_from_string_def(f,width_multiplier,block,
                                        force_identity_shortcut=(f.startswith('res') and prev_conv_output_num_of_channels==int(f[3:].split('x')[0])),
                                        return_sequences_rnn=return_sequences_rnn)(x)
        if f.startswith('lstm'):         
            prev_conv_output_num_of_channels=int(f[4:])
        elif f.startswith('res') or f.startswith('c1d'):
            prev_conv_output_num_of_channels=int(f[3:].split('x')[0])


    outputs = keras.layers.Dense(int(layer_strings[-1][1:])*width_multiplier, activation='linear')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=initial_lr), loss="mse")
    return model

def build_resnet_model(nhidden1=8, nhidden2=2, output_shape=1, act_func='sigmoid',
                       filters=num_sheets[data_file]-1, kernel_size=3, stride=1,
                       num_res_blocks=1):
    inputs = layers.Input(shape=NFEATURES* (ndays + nwindows))
    x = layers.Reshape((ndays+nwindows,NFEATURES))(inputs)
    for ii in range(num_res_blocks - 1):
        # TODO: think about conv filter numbers and kernel sizes
        intermediate_features = layers.ZeroPadding1D(padding=1,name="padding%d_branch2a" %ii)(x)
        intermediate_features = layers.Conv1D(filters=NFEATURES,kernel_size=2,strides=1,use_bias=False,
                          name="res%d_branch2a" %ii)(intermediate_features)
        intermediate_features = layers.BatchNormalization()(intermediate_features)
        intermediate_features = layers.Activation("relu", name="res%d_branch2a_relu" % ii)(intermediate_features)

        intermediate_features = layers.Conv1D(filters=NFEATURES,kernel_size=2,strides=1,use_bias=False,
                          name="res%d_branch2b" %ii)(intermediate_features)
        intermediate_features = layers.BatchNormalization()(intermediate_features)
        intermediate_features = layers.Activation("relu", name="res%d_branch2b_relu" % ii)(intermediate_features)

        shortcut = x
        x = layers.Add(name="res%d_add" % ii)([intermediate_features, shortcut])

    y = layers.ZeroPadding1D(padding=1,name="padding%d_branch2a" % num_res_blocks)(x)
    y = layers.Conv1D(filters,kernel_size,strides=stride,use_bias=False,
                                name="res%d_branch2a" % num_res_blocks)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu", name="res%d_branch2a_relu" % num_res_blocks)(y)

    y = layers.ZeroPadding1D(padding=1,name="padding%d_branch2b" % num_res_blocks)(y)
    y = layers.Conv1D(filters,kernel_size,use_bias=False,
                            name="res%d_branch2b" % num_res_blocks)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Flatten()(y)
    y = layers.Dense(nhidden1, activation=act_func)(y)

    shortcut = inputs
    shortcut = layers.Dense(nhidden1, activation=act_func)(shortcut)

    y = layers.Add(name="res%d_add" % num_res_blocks)([y, shortcut])
    
    y = layers.Activation("relu",name="res_relu")(y)


    y = layers.Dense(nhidden2, activation=act_func)(y)
    outputs= layers.Dense(output_shape, activation=keras.activations.linear,name='output')(y)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=initial_lr), loss=mse_loss_masked)
    return model

def build_residual_lstm_model(nhidden1=8, nhidden2=2, output_shape=1, act_func='sigmoid',rnn_layer_name='lstm1',conv_init=None):
    layer_type, rnn_side_branch_multiplier, _ = re.split('(\d+)',rnn_layer_name)
    rnn_layer = layers.LSTM if layer_type == 'lstm' else layers.GRU
    inputs = layers.Input(shape=NFEATURES*(ndays+nwindows))
    x = layers.Reshape((ndays+nwindows,num_sheets[data_file]-1))(inputs)
    x = layers.Permute((2,1))(x)

    y = tf.keras.layers.Conv1D(8+10,1, activation='relu',
                            kernel_initializer=conv_init,
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
                            trainable=False)(x)

    y = layers.Flatten()(y)
    y = layers.Dense(nhidden1, activation=act_func)(y)
    y = layers.Dense(nhidden2, activation=act_func)(y)
    y = layers.Dense(output_shape, activation=keras.activations.linear,name='mlp_output')(y)

    shortcut = x
    shortcut = layers.Dense(nhidden1, activation=act_func)(shortcut)
    shortcut = rnn_layer(units = output_shape*int(rnn_side_branch_multiplier), activation=act_func,return_sequences=True)(shortcut)
    shortcut = layers.Flatten()(shortcut)
    shortcut = layers.Dense(output_shape, activation=keras.activations.linear,name='lstm_output')(shortcut)

    outputs = layers.Add(name="res_add")([y, shortcut])
    # outputs = layers.Activation("relu",name="res_relu")(outputs)
    outputs = layers.LeakyReLU(alpha=0.3,name="res_relu")(outputs)
    
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=initial_lr), loss=mse_loss_masked)
    return model

"""## Train Models"""
def trainmodels(final_groups,group_stations):
    for group_name, stations in final_groups[group_stations].items():
        start = time.time()
        if not train_models:
            break
        # # prepare dataset
        selected_output_variables = []
        for station in stations:
            for output in output_locations:
                if station in output:
                    selected_output_variables.append(output)

        model_path_prefix = "mtl_%s_%s_%s_%s" % (group_name, full_model_str_def, num_dataset[data_file],str(int(train_frac*100))+'_train')

        print('Training model %s for %d stations: ' % (full_model_str_def, len(selected_output_variables)),end='')

        print([station.replace('target/','').replace('target','') for station in selected_output_variables])
        

        # create tuple of calibration and validation sets and the xscaler and yscaler on the combined inputs
        (xallc, yallc), (xallv, yallv), xscaler, yscaler = \
            annutils.create_training_sets([dfinps],
                                        [dfouts[selected_output_variables]],
                                        train_frac=train_frac,
                                        ndays=ndays,window_size=window_size,nwindows=nwindows)
        if model_str_def.startswith('resnet'):
            model = build_resnet_model(nhidden1=resnet_mult[0]*yallc.shape[1], nhidden2=resnet_mult[1]*yallc.shape[1], output_shape=yallc.shape[1],
                                    num_res_blocks=num_res_blocks)
        elif model_str_def.startswith('res'):
            conv_init = tf.constant_initializer(annutils.conv_filter_generator(ndays=8,window_size=11,nwindows=10))
            num_neurons_multiplier=[int(x) for x in model_str_def.split('_')[2:]]
            model = build_residual_lstm_model(num_neurons_multiplier[0]*len(output_locations),
                                    num_neurons_multiplier[1]*len(output_locations),
                                    output_shape=yallc.shape[1],
                                    act_func='sigmoid',
                                    rnn_layer_name=model_str_def.split('_')[1],
                                    conv_init=conv_init)
        else:
            model = build_model_from_string_def(full_model_str_def,width_multiplier=yallc.shape[1])

        (model.summary())
        history = model.fit(
            xallc,
            yallc,
            epochs=epochs,
            batch_size=128,
            validation_data=(xallv, yallv),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=50, mode="min", restore_best_weights=True),
                tensorboard_cb
            ],
            verbose=0,
        )
        # pd.DataFrame(history.history).hvplot(logy=True) # if you want to view the graph for calibration/validation training
        if save_results:
            annutils.save_model(os.path.join(gdrive_root_path, 'models', model_path_prefix) , model, xscaler, yscaler)

        print('Training time: %.2f seconds' % (time.time()-start))
if __name__ == "__main__":
    trainmodels(final_groups,group_stations)