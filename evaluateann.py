#%%
import os
import sys
#from typing_extensions import assert_type
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle
import time
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import holoviews as hv
hv.extension('bokeh')
import hvplot.pandas
import panel as pn
import annutils
from collections import defaultdict
#%%
from hyperparams import model_str_def, full_model_str_def, input_prepro_option, group_stations,\
                        exclude_stations, train_frac, window_size, epochs,\
                        train_models, save_results, percentiles, initial_lr,\
                        ndays, nwindows

from trainann import num_dataset, num_sheets, dflist, output_locations, data_file,\
                     gdrive_root_path, final_groups, mse_loss_masked
#%%
"""# Evaluation

## Define evaluation metrics
"""


eval_metrics = ['MSE', 'Bias', 'R', 'RMSD', 'NSE']

# key_stations = ['RSAC064','CCWDRock','CVPIntake', 'CCFBIntake','Emmaton','JerseyPoint']
key_stations = ['RSAC064','CCWD_Rock','CHDMC006', 'CHSWP003','RSAC092','RSAN018']
key_station_mappings = {'CCWD Rock': 'RSL',
                        # 'JerseyPoint':'RSAN018',
                        # 'Emmaton':'RSAC092', 
                        # 'CCFBIntake':'CHSWP003',
                        # 'CVPIntake':'CHDMC006',
                        # 'SLMZU025':'SLMZU025',
                        # 'ROLD059':'ROLD059',
                        'Middle River Intake':'MUP',
                        'Old River Hwy 4': 'OH4',
                        'OLD MID':'OLD_MID'}


def evaluate_sequences(target, pred, metrics,print_details=False):
    assert len(target) == len(pred), 'Target and predicted sequence length must equal.'
    valid_entries = target>0
    sequence_length = np.sum(valid_entries)
    # print('Total samples: %d, valid samples: %d' % (len(target), np.sum(valid_entries)))
    if np.any(sequence_length == 0):
        return {k: 0 for k in metrics}
    target=target[valid_entries]
    pred = pred[valid_entries]
    SD_pred = np.sqrt( np.sum((pred-np.mean(pred)) ** 2) /(sequence_length-1))
    SD_target = np.sqrt( np.sum((target-np.mean(target)) ** 2) /(sequence_length-1))

    eval_results = defaultdict(float)
    
    for m in metrics:
        if m =='MSE':
            eval_results[m] = ((target - pred)**2).mean()
        elif m =='Bias':
            eval_results[m] = np.sum(pred - target)/np.sum(target) * 100
        elif m == 'R':
            eval_results[m] = np.sum((pred-np.mean(pred)) * (target - np.mean(target))) / (sequence_length * SD_pred * SD_target)
        elif m == 'RMSD':
            eval_results[m] = np.sqrt(np.sum( ( ( pred-np.mean(pred) ) * ( target - np.mean(target) ) ) ** 2 ) / sequence_length)
        elif m == 'NSE':
            eval_results[m] = 1 - np.sum( ( target - pred ) ** 2 ) / np.sum( (target - np.mean(target) ) ** 2 )
    if print_details:
        print('(sum(pred - mean(pred)) x (target - mean(target))) =  %.4f' % np.sum((pred-np.mean(pred)) * (target - np.mean(target))))
        print('MSE =  %.4f' % eval_results[m])
        print('sum(pred - target) = %.4f' % np.sum(pred - target))
        print('sum(target) = %.4f' % np.sum(target))
        print('target standard deviation = %.6f, prediction standard deviation =%.6f' %(SD_target,SD_pred))
    return eval_results

"""## Compute numerical results"""

full_results={}
range_results=defaultdict(defaultdict)


df_inpout = pd.concat(dflist[0:(num_sheets[data_file])],axis=1).dropna(axis=0)
dfinps = df_inpout.loc[:,~df_inpout.columns.isin(dflist[num_sheets[data_file]-1].columns)]
#dfinps_test = pd.read_csv('scen1_inp.csv',index_col=0, parse_dates = ['date'])
dfouts = df_inpout.loc[:,df_inpout.columns.isin(dflist[num_sheets[data_file]-1].columns)]


#%%
def testANN(final_groups):
    for group_name, stations in final_groups[group_stations].items():
        # prepare dataset
        selected_output_variables = []
        for station in stations:
            for output in output_locations:
                if station in output:
                    selected_output_variables.append(output)
        print('Testing MTL ANN for %d stations: ' % len(selected_output_variables),end='')
        # print([station.replace('target/','').replace('target','') for station in selected_output_variables])

        model_path_prefix = "mtl_%s_%s_%s_%s" % (group_name, full_model_str_def, num_dataset[data_file],str(int(train_frac*100))+'_train')

        # create tuple of calibration and validation sets and the xscaler and yscaler on the combined inputs
        (xallc, yallc), (xallv, yallv), xscaler, yscaler = \
            annutils.create_training_sets([dfinps],
                                        [dfouts[selected_output_variables]],
                                        train_frac=train_frac,
                                        ndays=ndays,window_size=window_size,nwindows=nwindows)

        annmodel = annutils.load_model(os.path.join(gdrive_root_path,'models', model_path_prefix),custom_objects={"mse_loss_masked": mse_loss_masked})

        train_pred = annmodel.model.predict(xallc)
        test_pred = annmodel.model.predict(xallv)
        
        all_target = np.concatenate((yallc,yallv),axis=0)
        all_pred = np.concatenate((train_pred,test_pred),axis=0)

        for ii, location in enumerate(selected_output_variables):
            if any(exclude_s in location.lower() for exclude_s in exclude_stations):
                continue
            if any([k.lower() in location.lower() for k in key_stations]):
                print_details = True
            else:
                print_details = False
            print(location)
            ## training results
            # y = dfouts.loc[:,location].copy()
            # y[y<0] = float('nan')
            # train_dfyp = pd.concat([y, dfp.loc[:,location]],axis=1).dropna()

            # print('Training R^2 ', r2_score(train_dfyp_norm.iloc[:,0],train_dfyp_norm.iloc[:,1]))

            train_results = evaluate_sequences(yallc.iloc[:,ii],train_pred[:,ii], eval_metrics,
                                            print_details=print_details)
            train_results['R^2'] = r2_score(train_pred[:,ii], yallc.iloc[:,ii])
            full_results['%s_train' %location] = train_results


            # test_dfyp_norm = pd.concat([dfouts_norm.loc[:,location], dfp_test_norm.loc[:,location]],axis=1).dropna()
            # test_dfyp = pd.concat([y, dfp_test.loc[:,location]],axis=1).dropna()


            eval_results = evaluate_sequences(yallv.iloc[:,ii], test_pred[:,ii], eval_metrics,
                                                print_details=print_details)
            eval_results['R^2'] = r2_score(test_pred[:,ii], yallv.iloc[:,ii])

            full_results['%s_test' %location] = eval_results

            # test_dfyp.columns=['target','prediction']
            
            # test_dfyp_norm['percent_mark'] = (test_dfyp['target'] > test_dfyp['target'].quantile(0.25)).astype(int) * 25
            # test_dfyp_norm.loc[test_dfyp['target'] > test_dfyp['target'].quantile(0.75),'percent_mark']=75
            for (lower_quantile, upper_quantile) in zip(percentiles,percentiles[1:]+[1,]):
                if print_details:
                    print('#'*20, '%d%% - %d%%' % ((lower_quantile*100, upper_quantile*100)),'#'*20)
                lower_threshold = np.quantile(all_target[:,ii], lower_quantile)
                upper_threshold = np.quantile(all_target[:,ii], upper_quantile)
                eval_results = evaluate_sequences(all_target[(all_target[:,ii] > lower_threshold) & (all_target[:,ii] <= upper_threshold),ii],
                                                all_pred[(all_target[:,ii] > lower_threshold) & (all_target[:,ii] <= upper_threshold),ii],
                                                eval_metrics,
                                                print_details=print_details)
                range_results[location][lower_quantile*100] = eval_results

    if save_results:
        # create a pickle file on Google Drive and write results 
        result_path_prefix = "mtl_%s_%s_%s_%s" % (group_name, 'i%d_'%(ndays + nwindows) +model_str_def, num_dataset[data_file],str(int(train_frac*100))+'_train')
        results_path = os.path.join(gdrive_root_path,"results/%s_full_results.pkl" % (result_path_prefix))

        f = open(results_path,"wb")
        pickle.dump(full_results,f)
        f.close()
        print('Numerical results saved to %s' % results_path)


#%%
#def run_ann(final_groups,group_stations,selected_key_stations):
def run_ann(selected_key_station,dfinps):
    for group_name, stations in final_groups[group_stations].items():
        # prepare dataset
        selected_output_variables = []
        for station in stations:
            for output in output_locations:
                if station in output:
                    selected_output_variables.append(output)
        print('Testing MTL ANN for %d stations: ' % len(selected_output_variables),end='')

        print([station.replace('target/','').replace('target','') for station in selected_output_variables],end='\n')
        model_path_prefix = "mtl_%s_%s_%s_%s" % (group_name, 'i%d_'%(ndays + nwindows) + model_str_def, num_dataset[data_file],str(int(train_frac*100))+'_train')

        annmodel = annutils.load_model(os.path.join(gdrive_root_path,'models', model_path_prefix),custom_objects={"mse_loss_masked": mse_loss_masked})

        ## training results
        dfp=annutils.predict(annmodel.model, dfinps, annmodel.xscaler, annmodel.yscaler,columns=selected_output_variables,
                            ndays=ndays,window_size=window_size,nwindows=nwindows)

        y = dfouts.loc[:,selected_key_station].copy()
        y[y<0] = float('nan')
        targ_df = pd.DataFrame(y.iloc[(ndays+nwindows*window_size-1):])
        pred_df = pd.DataFrame(dfp.loc[:,selected_key_station])

        return targ_df,pred_df


#selected_key_stations = 'RSAC092-EMMATON'
#test_df = run_ann(selected_key_stations,dfinps_test)
#print(test_df)