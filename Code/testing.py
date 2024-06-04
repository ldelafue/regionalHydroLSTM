#for iii in range(22,25):
# -*- coding: utf-8 -*-
"""

@author: Luis De la Fuente
Final version based in Hydro_LSTM paper
"""
#%% Libraries
import argparse
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import *
from datetime import timedelta
from datetime import date
import random
import pickle
from sklearn.ensemble import RandomForestRegressor
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

#%% Functions
from HydroLSTM_global3 import *
from utils import *
#%% reading parameter
parser = argparse.ArgumentParser()
parser.add_argument('--country', choices=["US", "CL"])
parser.add_argument('--code', type=int)
parser.add_argument('--cells', type=int) 
parser.add_argument('--memory', type=int) 
parser.add_argument('--processor', default="cpu")
parser.add_argument('--model', choices=["HYDRO","HYDRO1reg", "HYDROglobal"])

cfg = vars(parser.parse_args()) 

country = cfg["country"]
code = cfg["code"]
cells = cfg["cells"]
memory = cfg["memory"]
model_option = cfg["model"]
processor = cfg["processor"]

# country = 'US'
# code = 1000000
# cells = 1
# memory = 512
# model_option = 'HYDROglobal'
# processor = 'cpu'

n_variables = 2 #ex. PP and PET = 2 variables
n_attributes = 17 

batch_size_values = [128]
dropout = 0
load_RF = True
out_of_sample = False

file_model = '1000000_C1_L512_hydroglobal_models.pkl'
file_RF_model = '1000000_C1_L512_hydroglobal_RF_model.pkl'
file_regression = '1000000_C1_L512_hydroglobal_RF_regression.pkl'
file_y_scaler = 'y_scaler_model_17.pkl'

if out_of_sample:
    list_catchments = 'out_sample.txt' 
else:
    list_catchments = 'list_IDs_all.csv' 

if code== 1000000:
    code_list =   pd.read_csv(list_catchments)
    code_att = pd.read_csv('gauge_information2.csv', index_col='GAGE_ID')
    code_att = code_att.loc[code_list.GAGE_ID,:]

    results = pd.DataFrame(columns=['lag', 'batch', 'cell', 'RMSE', 'R2', 'std_ratio', 'CC', 'Bias', 'KGE_testing'])
    results_catchment = pd.DataFrame(columns=['code','lag', 'batch', 'cell', 'RMSE', 'R2', 'std_ratio', 'CC', 'Bias_ratio', 'KGE_testing'])
    
    y_scaler_obs = pd.DataFrame(np.zeros([len(code_list),3]), columns=['y_max', 'y_min', 'y_mean'], index=code_list.GAGE_ID)

else:
    predictions = pd.DataFrame(np.zeros([9926 +2,22]))
    state_results = pd.DataFrame(np.zeros([9926 +2,21]))
    results = pd.DataFrame(columns=['lag', 'batch', 'cell', 'RMSE', 'R2', 'std_ratio', 'CC', 'Bias_ratio', 'KGE_testing'])

lag = memory + 1 
state_size = cells

x_max = []
x_min = []
x_mean = []
y_max = []
y_min = []
y_mean = []

warm_up = lag
input_size= n_variables*lag


if processor == "cpu":
    DEVICE = torch.device("cpu")
elif torch.backends.mps.is_available():                
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("gpu")
print(DEVICE)

if code == 1000000:

    x_max_i = []
    x_min_i = []
    x_mean_i = []
    y_max_i = []
    y_min_i = []
    y_mean_i = []
    q_list = []

    z=1
    for code_i in code_list.GAGE_ID:

        PP_i, PET_i, Q_i = load_data(code_i, country, warm_up) #warm_up ???

        
        if Q_i.isna().any().any():
            print('code ', code_i, 'has nan values')
            print(z)
            z = z+1
        else:
            ini_training = '1981/10/29' #'1981/10/01'
            training_last_day = '2000/09/30'
            validation_last_day = '2004/09/30'
            
            PP_train_i = PP_i[PP_i.index <= training_last_day]
            PET_train_i = PET_i[PET_i.index <= training_last_day] 
            Q_train_i = Q_i[Q_i.index <= training_last_day] 
            
            PP_test_i = PP_i.copy() 
            PET_test_i = PET_i.copy() 
            Q_test_i = Q_i.copy() 
            
            PP_valid_i = PP_i[PP_i.index <= validation_last_day] 
            PET_valid_i = PET_i[PET_i.index <= validation_last_day]
            Q_valid_i = Q_i[Q_i.index <= validation_last_day] 

            print('Q_i.Q_obs.min():',Q_i.Q_obs.min())

            Q_final = Q_test_i.copy()
            
            Q_final['PP'] = PP_test_i.PP.shift(lag)
            Q_final['PET'] = PET_test_i.PET.shift(lag)
            Q_final = Q_final.dropna()
            Q_final = Q_final.drop({'PP','PET'}, axis=1)
            Q_final = Q_final[Q_final.index >= ini_training]
            q_list.append(Q_final)
            
            if z==1:
              
                PP_train = PP_train_i.copy()
                PET_train = PET_train_i.copy()
                Q_train = Q_train_i.copy()
    
                PP_test = PP_test_i.copy()
                PET_test = PET_test_i.copy()
                Q_test = Q_test_i.copy()
                
                PP_valid = PP_valid_i.copy()
                PET_valid = PET_valid_i.copy()
                Q_valid = Q_valid_i.copy()
                
                
                ds = torch_dataset(PP_train,PET_train,Q_train, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=True) #create a class with x and y as torch dataset          
                x_max_i = ds.x_max
                x_min_i = ds.x_min
                x_mean_i = ds.x_mean
                y_max_i = ds.y_max
                y_min_i = ds.y_min
                y_mean_i = ds.y_mean
                
                y_scaler_obs.loc[code_i]['y_max'] = y_max_i
                y_scaler_obs.loc[code_i]['y_min'] = y_min_i
                y_scaler_obs.loc[code_i]['y_mean'] = y_mean_i
                
                att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values,code_i), dtype=torch.float32)
                att_tensor  = att_tensor.repeat(ds.x.shape[0], 1)
                ds.x = torch.cat((ds.x,att_tensor), 1)                         
                len_train = len(ds.y)
                
                
                ds_valid = torch_dataset(PP_valid,PET_valid, Q_valid, lag, ini_training, ds.x_max, ds.x_min, ds.x_mean, ds.y_max, ds.y_min, ds.y_mean, istrain=False)          
                att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values,code_i), dtype=torch.float32)
                att_tensor  = att_tensor.repeat(ds_valid.x.shape[0],1)
                ds_valid.x = torch.cat((ds_valid.x,att_tensor), 1)           
                len_valid = len(ds_valid.y)

                
                ### TESTING
                ds_full = torch_dataset(PP_test,PET_test,Q_test, lag, ini_training, ds.x_max, ds.x_min, ds.x_mean, ds.y_max, ds.y_min, ds.y_mean, istrain=False)
                att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values,code_i), dtype=torch.float32)
                att_tensor  = att_tensor.repeat(ds_full.x.shape[0],1)
                ds_full.x = torch.cat((ds_full.x,att_tensor), 1)                            
                len_test = len(ds_full.y) # - len_train - len_valid
                
            else:
                
                ds_i = torch_dataset(PP_train_i,PET_train_i,Q_train_i, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=True) #create a class with x and y as torch dataset          
                att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values,code_i), dtype=torch.float32)
                att_tensor  = att_tensor.repeat(ds_i.x.shape[0],1)
                ds_i.x = torch.cat((ds_i.x,att_tensor), 1) 
                len_train_i = len(ds_i.y)
                

                y_scaler_obs.loc[code_i]['y_max'] = ds_i.y_max 
                y_scaler_obs.loc[code_i]['y_min'] = ds_i.y_min
                y_scaler_obs.loc[code_i]['y_mean'] = ds_i.y_mean
                
                ds_valid_i = torch_dataset(PP_valid_i,PET_valid_i, Q_valid_i, lag, ini_training, ds_i.x_max, ds_i.x_min, ds_i.x_mean, ds_i.y_max, ds_i.y_min, ds_i.y_mean, istrain=False)          
                att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values,code_i), dtype=torch.float32)
                att_tensor  = att_tensor.repeat(ds_valid_i.x.shape[0],1)
                ds_valid_i.x = torch.cat((ds_valid_i.x,att_tensor), 1)  
                len_valid_i = len(ds_valid_i.y)
                
                ### TESTING
                ds_full_i = torch_dataset(PP_test_i,PET_test_i,Q_test_i, lag, ini_training, ds_i.x_max, ds_i.x_min, ds_i.x_mean, ds_i.y_max, ds_i.y_min, ds_i.y_mean, istrain=False)
                att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values,code_i), dtype=torch.float32)
                att_tensor  = att_tensor.repeat(ds_full_i.x.shape[0],1)
                ds_full_i.x = torch.cat((ds_full_i.x,att_tensor), 1)   
                len_test_i = len(ds_full_i.y)# - len_train_i - len_valid_i
                print('ds_full_i.y.numpy().min(): ',ds_full_i.y.numpy().min())
                
                
                ds.x = torch.cat((ds.x,ds_i.x),0)
                ds.y = torch.cat((ds.y,ds_i.y),0)
                ds.x = ds.x.float()
                
                ds_valid.x = torch.cat((ds_valid.x,ds_valid_i.x),0)
                ds_valid.y = torch.cat((ds_valid.y,ds_valid_i.y),0)
                ds_valid.x = ds_valid.x.float()
                
                ds_full.x = torch.cat((ds_full.x,ds_full_i.x),0)
                ds_full.y = torch.cat((ds_full.y,ds_full_i.y),0)
                ds_full.x = ds_full.x.float()
                

                
            print(z)
            z = z+1
            

    
    ds.x_max = ds.x.max(axis=0)#[0]
    ds.x_min = ds.x.min(axis=0)#[0]
    ds.x_mean = ds.x.mean(axis=0) #[-1,1]
    ds_valid.x_max = ds.x_max
    ds_valid.x_min = ds.x_min
    ds_valid.x_mean = ds.x_mean
    ds_full.x_max = ds.x_max
    ds_full.x_min = ds.x_min
    ds_full.x_mean = ds.x_mean

    ds.y_max = ds.y.max()
    ds.y_min = ds.y.min()
    ds.y_mean = ds.y.mean() #[-1,1]
    ds_valid.y_max = ds.y_max
    ds_valid.y_min = ds.y_min
    ds_valid.y_mean = ds.y_mean
    ds_full.y_max = ds.y_max
    ds_full.y_min = ds.y_min
    ds_full.y_mean = ds.y_mean            
 
    x_max = ds.x_max.values.numpy()
    x_min = ds.x_min.values.numpy()
    x_mean = ds.x_mean.numpy()
    y_max = ds.y_max.numpy()
    y_min = ds.y_min.numpy()
    y_mean = ds.y_mean.numpy()
    
    ds.num_samples = len(ds.y)
    ds_full.num_samples = len(ds_full.y)
    ds_valid.num_samples = len(ds_valid.y)
            
            

        
else:
    PP, PET, Q = load_data(code, country, warm_up)

    ini_training = '1981/10/29'
    training_last_day = '2000/09/30'
    validation_last_day = '2004/09/30'
    
    PP_train = PP[PP.index <= training_last_day]
    PET_train = PET[PET.index <= training_last_day] 
    Q_train = Q[Q.index <= training_last_day] 
    
    PP_test = PP.copy() 
    PET_test = PET.copy() 
    Q_test = Q.copy() 
    
    PP_valid = PP[PP.index <= validation_last_day] 
    PET_valid = PET[PET.index <= validation_last_day]
    Q_valid = Q[Q.index <= validation_last_day] 
                                   
    ds = torch_dataset(PP_train,PET_train,Q_train, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=True) #create a class with x and y as torch dataset          
    x_max = ds.x_max
    x_min = ds.x_min
    x_mean = ds.x_mean
    y_max = ds.y_max
    y_min = ds.y_min
    y_mean = ds.y_mean
    len_train = len(ds.y)
    print('training:',len_train)
    

    ds_valid = torch_dataset(PP_valid,PET_valid, Q_valid, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=False)          
    len_valid = len(ds_valid.y)
    ds_valid.num_samples = len(ds_valid.y)
    print('validation:',len_valid)
    
    ### TESTING
    ds_full = torch_dataset(PP_test,PET_test,Q_test, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=False)
    len_test = len(ds_full.y) # - len_train - len_valid
    print('testing:',len_test)


inputs_RF = code_att.iloc[:,-n_attributes:]

sampler_test = SequentialSampler(ds_full)
loader_test = DataLoader(ds_full, batch_size=1, sampler=sampler_test, shuffle=False)
        
# y_scaler_model = RandomForestRegressor(n_estimators = 200, bootstrap= True, max_depth= None, max_features=0.5, min_samples_split = 3, min_samples_leaf=2,  n_jobs=-1, oob_score=False)
# y_scaler_model.fit(inputs_RF, y_scaler)
# pickle.dump(y_scaler_model, open('y_scaler_model.pkl', 'wb'))


                        
if model_option == 'HYDRO':
    model = Model_hydro_lstm(input_size, lag, state_size, dropout).to(DEVICE)
elif model_option == 'HYDRO1reg':    
    model = Model_hydro_lstm_1reg(input_size, lag, state_size, dropout).to(DEVICE)
elif model_option == 'HYDROglobal':    
    model = Model_hydro_lstm_global(input_size, lag, state_size, dropout, n_attributes).to(DEVICE)
else:
    print('PENDING')


if load_RF:
    rf_model = pickle.load(open(file_RF_model, "rb" ))
    w_pred = rf_model.predict(inputs_RF)
    RF_weights = pd.DataFrame(data= w_pred, index=inputs_RF.index)

    rf_regression_model = pickle.load(open(file_regression, "rb" ))                
    regression_pred = rf_regression_model.predict(inputs_RF)
    RF_regression = pd.DataFrame(data= regression_pred, index=inputs_RF.index)


    model = pickle.load(open(file_model, "rb" )) 

if out_of_sample:
    
    y_scaler_model = pickle.load(open(file_y_scaler, "rb" ))                
    y_scaler = y_scaler_model.predict(inputs_RF)
    y_scaler = pd.DataFrame(data= y_scaler, index=inputs_RF.index)
    y_scaler.rename(columns={0: 'y_mean', 1: 'y_max', 2: 'y_min'},inplace=True)
    y_scaler['y_max'] = y_scaler.y_max *  (inputs_RF.pet_mean/inputs_RF.p_mean)
    y_scaler['y_mean'] = y_scaler.y_mean *  (inputs_RF.pet_mean/inputs_RF.p_mean)
    
else:
    y_scaler = y_scaler_obs



iteration = 1 
for data in loader_test:

    x_epoch, y_epoch = data
   
    catch_ID =  int(np.round(x_epoch[0,-1].detach().numpy(),decimals=0))
    
    x_epoch, y_epoch = x_epoch.to(DEVICE), y_epoch.to(DEVICE)
    y_epoch = y_epoch.resize(len(y_epoch),1) 
    
    if load_RF:
        pred_epoch = model(x_epoch,RF_weights, RF_regression)[0]
    else:
        pred_epoch = model(x_epoch)[0]
    
    
    
    pred_epoch = pred_epoch.cpu().detach().numpy()

    
    c_epoch = model.c_t
    c_epoch = c_epoch.cpu().detach().numpy()
          
    y_epoch = y_epoch.cpu().detach().numpy() 


    if load_RF:
        pred_epoch = (pred_epoch*(y_scaler.loc[catch_ID]['y_max'] - y_scaler.loc[catch_ID]['y_min']) + y_scaler.loc[catch_ID]['y_mean'])
        y_epoch = (y_epoch*(y_scaler_obs.loc[catch_ID]['y_max'] - y_scaler_obs.loc[catch_ID]['y_min']) + y_scaler_obs.loc[catch_ID]['y_mean'])
        
    else:
        pred_epoch = pred_epoch*(y_max - y_min) + y_mean
        y_epoch = y_epoch*(y_max - y_min) + y_mean    
       
    if iteration == 1:
        q_sim = pred_epoch
        q_obs = y_epoch
        state_value = c_epoch[1:]
    else:
        q_sim = np.concatenate((q_sim, pred_epoch))
        q_obs = np.concatenate((q_obs, y_epoch))
        state_value = np.concatenate((state_value, c_epoch[1:]))
        #print(c_epoch)
    iteration = iteration + 1
    




if code == 1000000:
    i =0            
    l_period = int(len(q_sim)/len(code_list))

    #q_sim = q_sim*(y_max - y_min) + y_mean
    #q_obs = q_obs*(y_max - y_min) + y_mean

    q_sim = q_sim.flatten() #**power_exp) + neg_value.numpy()
    q_obs = q_obs.flatten() #**power_exp) + neg_value.numpy()
    state_value = state_value.flatten()
    
    #ValueError: Length of values (924918) does not match length of index (919979)
    Q_final = pd.concat(q_list)
    #Q_final = Q_final[Q_final.index >= ini_training]
    Q_final['q_sim'] = q_sim
    Q_final['q_obs'] = q_obs
    Q_final['state_value'] = state_value
    
    #Q_final = Q_final.set_index('basin', append=True)
    #print('neg_value: ', neg_value)    
    nan_indices = np.isnan(q_obs)
    
    RMSE = mean_squared_error(q_sim[~nan_indices], q_obs[~nan_indices])**0.5
    MAE = mean_absolute_error(q_sim[~nan_indices], q_obs[~nan_indices])
    R2 = r2_score(q_sim[~nan_indices], q_obs[~nan_indices])
    
    BIAS = q_sim[~nan_indices].sum() / q_obs[~nan_indices].sum()
    CC = np.corrcoef([q_sim[~nan_indices], q_obs[~nan_indices]],rowvar=True)
    CC = CC[0,1]
    mean_s = q_sim[~nan_indices].mean()
    mean_o = q_obs[~nan_indices].mean()
    std_s = q_sim[~nan_indices].std()
    std_o = q_obs[~nan_indices].std()  
    KGE = 1 - ((CC - 1) ** 2 + (std_s / std_o - 1) ** 2 + (mean_s / mean_o - 1) ** 2) ** 0.5
    
    results.at[i,'lag'] = lag-1
    results.at[i,'batch'] =  len(y_epoch)
    results.at[i,'cell'] = state_size
    results.at[i,'RMSE'] = RMSE
    results.at[i,'R2'] = R2
    results.at[i,'std_ratio'] = std_s / std_o
    results.at[i,'CC'] = CC
    results.at[i,'Bias'] = BIAS  
    results.at[i,'KGE_testing'] = KGE 
    print(results)

    for ii in range(len(code_list)):
        
        q_sim_i = Q_final[Q_final.basin == code_list.GAGE_ID[ii]].q_sim.values
        q_obs_i = Q_final[Q_final.basin == code_list.GAGE_ID[ii]].q_obs.values
        state_value_i = Q_final[Q_final.basin == code_list.GAGE_ID[ii]].state_value.values
        
        
        q_sim_i_period = q_sim_i[-(len_test_i-len_valid_i):]
        q_obs_i_period = q_obs_i[-(len_test_i-len_valid_i):]
        nan_indices = np.isnan(q_obs_i_period)
        q_sim_i_period_nan = q_sim_i_period[~nan_indices]
        q_obs_i_period_nan = q_obs_i_period[~nan_indices]        

        RMSE_ii = mean_squared_error(q_sim_i_period_nan, q_obs_i_period_nan)**0.5
        MAE_ii = mean_absolute_error(q_sim_i_period_nan, q_obs_i_period_nan)
        R2_ii = r2_score(q_sim_i_period_nan, q_obs_i_period_nan)
        BIAS_ii = q_sim_i_period_nan.sum() / q_obs_i_period_nan.sum()
        CC_ii = np.corrcoef([q_sim_i_period_nan, q_obs_i_period_nan],rowvar=True)
        CC_ii = CC_ii[0,1]
        mean_s_ii = q_sim_i_period_nan.mean()
        mean_o_ii = q_obs_i_period_nan.mean()
        std_s_ii = q_sim_i_period_nan.std()
        std_o_ii = q_obs_i_period_nan.std()  
        KGE_ii = 1 - ((CC_ii - 1) ** 2 + (std_s_ii / std_o_ii - 1) ** 2 + (mean_s_ii / mean_o_ii - 1) ** 2) ** 0.5
        
        results_catchment.at[i*len(code_list) + ii,'code'] = code_list.GAGE_ID[ii] 
        results_catchment.at[i*len(code_list) + ii,'lag'] = lag - 1
        results_catchment.at[i*len(code_list) + ii,'batch'] = len(y_epoch)
        results_catchment.at[i*len(code_list) + ii,'cell'] = state_size
        results_catchment.at[i*len(code_list) + ii,'RMSE'] = RMSE_ii
        results_catchment.at[i*len(code_list) + ii,'R2'] = R2_ii
        results_catchment.at[i*len(code_list) + ii,'std_ratio'] = std_s_ii / std_o_ii
        results_catchment.at[i*len(code_list) + ii,'CC'] = CC_ii
        results_catchment.at[i*len(code_list) + ii,'Bias_ratio'] = BIAS_ii
        results_catchment.at[i*len(code_list) + ii,'KGE_testing'] = KGE_ii


    dates = pd.date_range(start='1981-10-29', end='2008-12-31') #2000/09/30' 2004/09/30' '1981/10/29'
    list_dates = [dates] * len(code_list)
    concatenated_dates = []
    for idx in list_dates:
        concatenated_dates.extend(idx.tolist())

    expanded_list = []
    for item in code_list.GAGE_ID.values:
        expanded_list.extend([item] * l_period)
    
    predictions = pd.DataFrame(np.zeros([len(Q_final) + 2,4]))
    state_results = pd.DataFrame(np.zeros([len(Q_final) +2 ,3]))
    
    predictions.loc[0,1] = lag-1
    predictions.loc[1,1] = state_size
    predictions.loc[1,2] = 'ID'
    predictions.loc[1,0] = 'obs'
    predictions.loc[2:len(Q_final)+ 2,2] = Q_final.basin.values
    predictions.loc[2:len(q_obs)+ 2,0] = q_obs
    predictions.loc[2:len(q_sim)+ 2,1] = q_sim
    predictions.loc[2:len(Q_final)+ 2,3] = Q_final.index.astype(str)

    
    if state_size == 1:                    
        state_results.loc[0,0] = lag-1
        state_results.loc[1,0] = state_size
        state_results.loc[1:,1] = 'ID'
        state_results.loc[2:len(state_value)+ 2,0] = state_value
        state_results.loc[2:len(Q_final)+ 2,1] = Q_final.basin.values
        state_results.loc[2:len(Q_final) + 2,2] =Q_final.index.astype(str)
    
    print(results_catchment)

    
else:
    


    q_sim = q_sim.flatten()
    q_obs = q_obs.flatten()
    state_value = state_value.flatten()
               
    RMSE = mean_squared_error(q_sim[-len_test:], q_obs[-len_test:])**0.5
    MAE = mean_absolute_error(q_sim[-len_test:], q_obs[-len_test:])
    R2 = r2_score(q_sim[-len_test:], q_obs[-len_test:])
    
    CC = np.corrcoef([q_sim[-len_test:], q_obs[-len_test:]],rowvar=True)
    CC = CC[0,1]
    mean_s = q_sim[-len_test:].mean()
    mean_o = q_obs[-len_test:].mean()
    BIAS = mean_s / mean_o
    std_s = q_sim[-len_test:].std()  
    std_o = q_obs[-len_test:].std()  
    KGE = 1 - ((CC - 1) ** 2 + (std_s / std_o - 1) ** 2 + (mean_s / mean_o - 1) ** 2) ** 0.5
    
    results.at[i,'lag'] = lag-1
    results.at[i,'batch'] =  batch_size
    results.at[i,'cell'] = state_size
    results.at[i,'RMSE'] = RMSE
    results.at[i,'R2'] = R2
    results.at[i,'std_ratio'] = std_s / std_o
    results.at[i,'CC'] = CC
    results.at[i,'Bias'] = BIAS  
    results.at[i,'KGE_testing'] = KGE                   
    
    print(results)
    predictions.loc[2:,0] = Q_test.index[-(len_train + len_valid + len_test):].values
    predictions.loc[0,i+2] = lag-1
    predictions.loc[1,i+2] = state_size
    predictions.loc[2:,i+2] = q_sim
    predictions.loc[2:,1] = q_obs
    if state_size == 1:
        state_results.loc[2:,0] = Q_test.index[-(len_train + len_valid + len_test):].values
        state_results.loc[0,i+1] = lag-1
        state_results.loc[1,i+1] = state_size
        state_results.loc[2:,i+1] = state_value[-(len_train + len_valid + len_test):]              
    



            
if model_option == "HYDRO":
    if out_of_sample:
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_summary_testing_out.csv'
        results.to_csv(name_file)
        
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_predictions_testing_out.csv'
        predictions = pd.DataFrame(predictions)
        predictions.to_csv(name_file)
        
    
        if state_size == 1:
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_state_testing_out.csv'
            state_results = pd.DataFrame(state_results)
            state_results.to_csv(name_file)   
    
    else:    
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_summary_testing.csv'
        results.to_csv(name_file)
        
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_predictions_testing.csv'
        predictions = pd.DataFrame(predictions)
        predictions.to_csv(name_file)
        
    
        if state_size == 1:
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_state_testing.csv'
            state_results = pd.DataFrame(state_results)
            state_results.to_csv(name_file)        

else:
    if out_of_sample:
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_' + 'hydroglobal_summary_testing_out.csv'
        results.to_csv(name_file)
        
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_' + 'hydroglobal_predictions_testing_out.csv'
        predictions = pd.DataFrame(predictions)
        predictions.to_csv(name_file)   
        
        if state_size == 1:
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_' + 'hydroglobal_state_testing_out.csv'
            state_results = pd.DataFrame(state_results)
            state_results.to_csv(name_file) 
            
        if code == 1000000:
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydroglobal_summary_per_catchment_testing_out.csv'
            results_catchment.to_csv(name_file)
        
        if load_RF:
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydroglobal_weights_testing_out.csv'
            RF_weights.to_csv(name_file)
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydroglobal_regression_testing_out.csv'
            RF_regression.to_csv(name_file)
    else:
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_' + 'hydroglobal_summary_testing.csv'
        results.to_csv(name_file)
        
        name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_' + 'hydroglobal_predictions_testing.csv'
        predictions = pd.DataFrame(predictions)
        predictions.to_csv(name_file)   
        
        if state_size == 1:
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_' + 'hydroglobal_state_testing.csv'
            state_results = pd.DataFrame(state_results)
            state_results.to_csv(name_file) 
            
        if code == 1000000:
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydroglobal_summary_per_catchment_testing.csv'
            results_catchment.to_csv(name_file)
        
        if load_RF:
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydroglobal_weights_testing.csv'
            RF_weights.to_csv(name_file)
            name_file = str(code) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydroglobal_regression_testing.csv'
            RF_regression.to_csv(name_file)

