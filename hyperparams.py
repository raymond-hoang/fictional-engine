####################################################
############## Hyper-param definitions #############
####################################################

'''
Define the model architecture: (input layer will be build automatically)

Supported abbreviation-layer pairs (detailed definitions can be found in "Layer builders" section)
- 'lstm': keras.layers.LSTM
- 'res': resnet block (basic_1d)
- 'c1d': keras.layers.Conv1D
- 'td': keras.layers.TimeDistributed
- 'dr': keras.layers.Dropout
- 'f': keras.layers.Flatten
- 'g': keras.layers.GRU
- 'd': keras.layers.Dense 
- 'o': keras.layers.Dense

Usage of resnet blocks: res(num_of_filters)x(kernel_size)x(stride)x(stages)
stages: # of resnet units in the block
example: model_str_def = 'res10x3x1x1_f_d8_d2_o1'

Usage of 1D conv layers: c1d(num_of_filters)x(kernel_size)x(stride)
example: model_str_def = 'c1d10x3x1_c1d10x3x1_f_d8_d2_o1'

'''
# Note: numbers following 'd' or 'o' will be multiplied by [size of output] before being used as numbers of neurons in dense layer

''' 
4 architectures are available:
MLP; LSTM; GRU; ResNet
'''
## 1. To train an MLP Network: (uncomment both 2 lines below):
model_str_def = 'd8_d4_o1'
input_prepro_option=1

# 2. To train an LSTM Network (uncomment both 2 lines below):
# model_str_def = 'lstm8_lstm4_f_o1'
# model_str_def = 'lstm14_f_o1'
# input_prepro_option = 2


## 3. To train a GRU Network (uncomment both 2 lines below):
# model_str_def = 'g14_f_o1'
# input_prepro_option = 2

# ## 4. To train a ResNet (uncomment both 2 lines below):
# model_str_def = 'resnet_16_8'
# num_res_blocks=1
# resnet_mult = [int(x) for x in model_str_def.split('_')[1:]]
# input_prepro_option = 2

# # 5. To train a Res-LSTM (uncomment both lines below):
# model_str_def = 'res_lstm2_8_4' # original res_lstm = 'res_lstm2_8_2'
# input_prepro_option = 2

# # 6. To train a Res-GRU (uncomment both lines below):
# model_str_def = 'res_gru2_8_6' # original res_gru = 'res_gru2_8_2'
# input_prepro_option = 2


''' 
Grouping stations or not?
'''
# if group_stations == True, 29 stations will be split into 3 groups
# if group_stations == False, 29 stations will be considered as 1 group
group_stations = False
exclude_stations = ['martinez', 'chvct000']

''' 
Fraction of training set
'''
train_frac = 0.7


'''
Define parameters for input pre-processing
- ndays: number of daily values in inputs
- window_size: length of averaging windows
- nwindows: number of moving averages
'''
if input_prepro_option ==1:
    # option 1: apply pre-defined average windowing:
    ndays=8
    window_size=11
    nwindows=10
elif input_prepro_option ==2:
    # option 2: directly use daily measurements as inputs
    ndays=118
    window_size=0
    nwindows=0
else:
    raise "input_prepro_option=%d is not supported, please select between {0, 1}" % input_prepro_option


# number of training epochs
# training will stop when reaching this number or test loss doesn't decrease for 50 epochs
epochs = 5000


# set to False if you just want to evaluate trained models
train_models = True
save_results = True
full_model_str_def = 'i%d_'%(ndays + nwindows) +model_str_def

####################################################
########## End of hyper-param definitions ##########
####################################################

# "dsm2_ann_inputs_20220204.xlsx": simulated dataset
# "dsm2_ann_inputs_20220215.xlsx": observed dataset at daily resolution
# "dsm2_ann_observed_15min.xlsx":  observed dataset at 15-min resolution



# percentile thresholds for ranged results
percentiles = [0,0.75,0.95]  

initial_lr=0.001
