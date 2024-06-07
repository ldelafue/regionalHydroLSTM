"""
@author: Luis De la Fuente
"""
#%% Libraries

import argparse 
from typing import Tuple # Import the Tuple type hint from the typing module
# Import the PyTorch library and its submodules
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
# Import tqdm for progress bar visualization
from tqdm import tqdm 
import pandas as pd 
import sys 
import matplotlib.pyplot as plt
from sklearn.metrics import * 
from datetime import timedelta 
from datetime import date
from datetime import datetime
import random 
import pickle
import os 
from sklearn.ensemble import RandomForestRegressor 
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

#notes
#MPI: CPU

#%% Functions
from Hydro_LSTM import *
#from HydroLSTM_regional import *
from HydroLSTM_global3 import *
from utils import *
#%% reading parameter
parser = argparse.ArgumentParser()  # Create an ArgumentParser object to handle command-line arguments
parser.add_argument('--code', type=int, help='Gauge ID of the catchment trained with HydroLSTM')  
parser.add_argument('--memory', type=int,help='Number of lagged days used for HydroLSTM (Zero is the current time step)')
parser.add_argument('--cells', type=int, help='Number of HydroLSTM cells in parallel (only for HydroLSTM)')    
parser.add_argument('--model', choices=["HYDRO","regionalHYDRO"], help='Local or regional HydroLSTM to use in the training')  

cfg = vars(parser.parse_args())  # Parse the command-line arguments and convert them to a dictionary

# cfg["memory"] = 512
# cfg["model"] = "HYDRO"
# cfg["cells"] = int(1)
# cfg["learning_rate"] = 0.0001
# cfg["code"] = 11230500
# cfg["epochs"] = int(5)

#%%
#Initialization of general parameters
initial_epoch = 0
patience = 0  # Set the patience parameter equal to the number of epochs
n_variables = 2  # Number of variables (e.g., Precip and PET) is set to 2
n_attributes = 17  # Number of attributes used from CAMELS dataset
dropout = 0  # Set the dropout rate to 0. Parsimonious model need all the conections
processor = "cpu"  # Set the processor to "cpu"
batch_size_values = [1]  # Define batch size values as a list with a single element 128
n_models_HydroLSTM = 1 #number of random models run only for HydroLSTM 
load_RF = True  # Set a flag to indicate whether to load the initial weights requiered to run Random Forest model
first_RF_epoch = 4
RF_step = 20
RF_plot = False
path_RF_weight = '1000000_C1_L512_regionalhydro_weights_base.csv'


#%% Setting an internal ID=1000000 when we train a regional model
if cfg["model"] == "regionalHYDRO":
    cfg["code"] = 1000000
    cfg["cells"] = int(1)
    cfg["memory"] = 512
    cfg["batch"] = 128
    batch_size_values = [cfg["batch"]]  

    # Get the current working directory
    current_directory = os.getcwd()
    
    # Change to the parent directory
    parent_directory = os.path.dirname(current_directory)
    
    # Change to the target directory within the parent directory
    target_directory = os.path.join(parent_directory, 'Results/RF_mean')
    path_RF_weight = target_directory + "/" + path_RF_weight
else:
    cfg["cells"] = int(1)

#%%
# Set initial training, last training day, and last validation day based on the code
if cfg["code"] == 1000000:
    ini_training = '1981/10/29' 
    training_last_day = '2000/09/30'
    validation_last_day = '2004/09/30'
else:
    ini_training = '1981/10/01'
    training_last_day = '2000/09/30'
    validation_last_day = '2004/09/30'

# Set the device based on processor availability
if processor == "cpu":
    DEVICE = torch.device("cpu")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("gpu")
print(DEVICE)
        
#%%  
# Creating emphy dataset

from datetime import datetime
#Calculating the number of days in the training and validation period
days_train_valid = (datetime.strptime('2008/12/31', '%Y/%m/%d') - datetime.strptime(ini_training, '%Y/%m/%d')).days + 1 
    
# Initialize data structures based on the provided code value
if cfg["code"] == 1000000:
    # Load gauge IDs and attributes
    code_list = pd.read_csv('list_IDs_all.csv')
    code_att = pd.read_csv('gauge_information2.csv', index_col='GAGE_ID')
    code_att = code_att.loc[code_list.GAGE_ID, :]


    
    # Initialize DataFrames for predictions, state results, and evaluation metrics for all gauges
    predictions = pd.DataFrame(np.zeros([days_train_valid * len(code_list) + 2, 4])) 
    state_results = pd.DataFrame(np.zeros([days_train_valid * len(code_list) + 2, 3]))
    results = pd.DataFrame(columns=['lag', 'batch', 'cell', 'RMSE', 'R2', 'std_ratio', 'CC', 'Bias', 'KGE_test'])
    results_catchment = pd.DataFrame(columns=['code', 'lag', 'batch', 'cell', 'RMSE', 'R2', 'std_ratio', 'CC', 'Bias_ratio', 'KGE_testing'])

else:
    # Initialize DataFrames for predictions, state results, and evaluation metrics for a single gauge
    predictions = pd.DataFrame(np.zeros([days_train_valid +2, n_models_HydroLSTM + 2 ]))
    state_results = pd.DataFrame(np.zeros([days_train_valid +2, n_models_HydroLSTM + 1]))
    results = pd.DataFrame(columns=['lag', 'batch', 'cell', 'RMSE', 'R2', 'std_ratio', 'CC', 'Bias_ratio', 'KGE_testing'])
    code_list = pd.DataFrame()
    code_att = pd.DataFrame()

# Convert the entire DataFrame to object dtype
predictions = predictions.astype(object)
state_results = state_results.astype(object)
#%%  
# Preparing dataset

# Determine state size values based on the chosen model
if cfg["model"] == "HYDRO":
    state_size_values = np.ones(n_models_HydroLSTM, dtype=int) 
    state_size_values = cfg["cells"] * state_size_values #[cfg["cells"] * s for s in state_size_values]
else:
    state_size_values = [cfg["cells"]]

# Initialize model_summary list and set initial values
model_summary = []
i = 0
lag = cfg["memory"] + 1 #adding the 0 day 

# Loop over batch size and state size values to generate datasets and model summaries
for batch_size in batch_size_values:
    for state_size in state_size_values:
        print('--------------------------------------------------------------------------------------------------------------')
        print(f'model #{i}')
        # Initialize lists for normalization statistics
        # x_max = []
        # x_min = []
        # x_mean = []
        # y_max = []
        # y_min = []
        # y_mean = []
        
        # Set warm-up period and input size
        warm_up = lag
        input_size = n_variables * lag

        # Create datasets for training, validation, and testing
        print('Preparing the dataset')
        ds, ds_valid, ds_full, y_scaler, len_train, len_valid , len_test = creating_dataset(cfg["code"], code_list, code_att, n_attributes,lag, ini_training, training_last_day, validation_last_day)
         

#%%  
# Charging the model and loader  
        
        # Get the current working directory
        current_directory = os.getcwd()
        
        # Change to the parent directory
        parent_directory = os.path.dirname(current_directory)
                    
        # Initialize and instantiate the model based on the model_option
        if cfg["model"] == 'HYDRO':

            # Change to the target directory within the parent directory
            target_directory = os.path.join(parent_directory, 'Results/hydroLSTM/best model')
            path_best_Hydro_model = target_directory + "/" + str(cfg["code"]) + '_C1_L' + str(cfg["memory"]) + '_hydro_best_model.pkl'
            
            # Opening the best model
            model = pickle.load(open(path_best_Hydro_model, "rb"))

        elif cfg["model"] == 'regionalHYDRO':

            # Change to the target directory within the parent directory
            target_directory = os.path.join(parent_directory, 'Results/RF_mean')
            path_Hydro_model = target_directory + "/" + '1000000_C1_L512_regionalhydro_model.pkl'
            path_RF_model = target_directory + "/" + '1000000_C1_L512_regionalhydro_RF_model.pkl'
            path_RF_regression = target_directory + "/" + '1000000_C1_L512_regionalhydro_RF_regression.pkl'

            # Opening the model
            model = pickle.load(open(path_Hydro_model, "rb"))
            rf_model = pickle.load(open(path_RF_model, "rb"))
            rf_regression_model = pickle.load(open(path_RF_regression, "rb"))
            
        
            
            
            
            
        else:
            print('Error in the name of the model')
            break
        
        # creating the loader
        if cfg["model"] == "HYDRO":
            sampler_test = SequentialSampler(ds_full)
            loader_test = DataLoader(ds_full, batch_size=ds_full.num_samples, sampler=sampler_test, shuffle=False)



        # training based on the model_option
        if cfg["model"] == 'regionalHYDRO':
                       
            # Create sampler and data loader for testing set
            sampler_test = SequentialSampler(ds_full)
            loader_test = DataLoader(ds_full, batch_size=len_test, sampler=sampler_test, shuffle=False)
            
  
            # Load random forest weights if specified
            if load_RF:

                RF_weights = pd.read_csv(path_RF_weight,  index_col= 0)    
                RF_weights = RF_weights.reindex(code_att.index)
                RF_weights.dropna(inplace=True)
                
                RF_regression = pd.DataFrame(index= RF_weights.index)
                RF_regression.at[:,0] = 1.0
                RF_regression.at[:,1] = 0.0
            


            
            # Defining inputs for RF model
            inputs_RF = code_att.iloc[:, -n_attributes:]

            #evaluating the weights from RF models
            w_pred = rf_model.predict(inputs_RF)
            RF_weights = pd.DataFrame(data=w_pred, index=inputs_RF.index)

            regression_pred = rf_regression_model.predict(inputs_RF)
            RF_regression = pd.DataFrame(data=regression_pred, index=inputs_RF.index)

                    





#%%  
# inference                

        # validation based on the model_option
        if cfg["model"] == 'regionalHYDRO':
                    
            # Initialize iteration counter
            iteration = 1
            
            # Iterate over data batches in the validation loader
            for data in loader_test:
            
                # Extract input features (x_epoch) and target labels (y_epoch) from the data batch
                x_epoch, y_epoch = data
            
                # Extract catchment ID from the input features
                catch_ID = int(np.round(x_epoch[0, -1].detach().numpy(), decimals=0))
            
                # Move data to the device (e.g., GPU) if available
                x_epoch, y_epoch = x_epoch.to(DEVICE), y_epoch.to(DEVICE)
                y_epoch.resize_(len(y_epoch), 1) 
            
                # Generate predictions for the current epoch
                pred_epoch = model(x_epoch, RF_weights, RF_regression)[0]

                # Convert predictions, target labels, and state values to numpy arrays
                pred_epoch = pred_epoch.cpu().detach().numpy()
                c_epoch = model.c_t.cpu().detach().numpy()
                y_epoch = y_epoch.cpu().detach().numpy()
            
                # Rescale predictions and target labels 
                pred_epoch = (pred_epoch * (y_scaler.loc[catch_ID]['y_max'] - y_scaler.loc[catch_ID]['y_min']) + y_scaler.loc[catch_ID]['y_mean'])
                y_epoch = (y_epoch * (y_scaler.loc[catch_ID]['y_max'] - y_scaler.loc[catch_ID]['y_min']) + y_scaler.loc[catch_ID]['y_mean'])
            
                # Concatenate predictions and target labels across iterations
                if iteration == 1:
                    q_sim = pred_epoch
                    q_obs = y_epoch
                    state_value = c_epoch[1:]
                else:
                    q_sim = np.concatenate((q_sim, pred_epoch))
                    q_obs = np.concatenate((q_obs, y_epoch))
                    state_value = np.concatenate((state_value, c_epoch[1:]))
            
                # Increment iteration counter
                iteration = iteration + 1
                
        # validation based on the model_option
        if cfg["model"] == 'HYDRO':
            iteration = 1
            for data in loader_test:
    
                x_epoch, y_epoch = data
    
                x_epoch, y_epoch = x_epoch.to(DEVICE), y_epoch.to(DEVICE)
                y_epoch.resize_(len(y_epoch),1)
    
                pred_epoch = model(x_epoch)[0]
                c_epoch = model(x_epoch)[2]
                c_epoch = c_epoch.cpu().detach().numpy()
    
                pred_epoch = pred_epoch.cpu().detach().numpy()
                y_epoch = y_epoch.cpu().detach().numpy() #
                if iteration == 1:
                    q_sim = pred_epoch
                    q_obs = y_epoch
                    if state_size == 1:
                        state_value = c_epoch[1:]
                else:
                    q_sim = np.concatenate((q_sim, pred_epoch))
                    q_obs = np.concatenate((q_obs, y_epoch))
                    if state_size == 1:
                        state_value = np.concatenate((state_value, c_epoch[1:]))
                iteration = iteration + 1



#%%  
# calculation of the metrics         

        if cfg["model"] == 'regionalHYDRO':
            
            # Calculate the length of each period based on the length of q_sim and the number of catchments
            l_period = int(len(q_sim) / len(code_list))
        
            # Flatten the arrays for calculation
            q_sim = q_sim.flatten()
            q_obs = q_obs.flatten()
            state_value = state_value.flatten()
        
            # Calculate evaluation metrics for the entire dataset
            RMSE = mean_squared_error(q_sim, q_obs) ** 0.5
            MAE = mean_absolute_error(q_sim, q_obs)
            R2 = r2_score(q_sim, q_obs)
            BIAS = q_sim.sum() / q_obs.sum()
            CC = np.corrcoef([q_sim, q_obs], rowvar=True)[0, 1]
            mean_s = q_sim.mean()
            mean_o = q_obs.mean()
            std_s = q_sim.std()
            std_o = q_obs.std()
            KGE = 1 - ((CC - 1) ** 2 + (std_s / std_o - 1) ** 2 + (mean_s / mean_o - 1) ** 2) ** 0.5
        
            # Update the results DataFrame with evaluation metrics for the entire dataset
            results.at[i, 'lag'] = lag - 1
            results.at[i, 'batch'] = batch_size
            results.at[i, 'cell'] = state_size
            results.at[i, 'RMSE'] = RMSE
            results.at[i, 'R2'] = R2
            results.at[i, 'std_ratio'] = std_s / std_o
            results.at[i, 'CC'] = CC
            results.at[i, 'Bias'] = BIAS
            results.at[i, 'KGE_valid'] = KGE
            
            end_train = ds_valid.num_samples
            
            # Iterate over catchments to calculate evaluation metrics for each catchment
            for ii in range(len(code_list)):
                q_sim_i = q_sim[ii * l_period: (ii + 1) * l_period]
                q_obs_i = q_obs[ii * l_period: (ii + 1) * l_period]
                state_value_i = state_value[ii * l_period: (ii + 1) * l_period]
                len_valid_i = int(ds_full.num_samples/len(code_list))
                len_train = int(ds_valid.num_samples/len(code_list))
                
                # Calculate evaluation metrics for each catchment
                RMSE_ii = mean_squared_error(q_sim_i[-(len_valid_i - len_train):], q_obs_i[-(len_valid_i - len_train):]) ** 0.5
                MAE_ii = mean_absolute_error(q_sim_i[-(len_valid_i - len_train):], q_obs_i[-(len_valid_i - len_train):])
                R2_ii = r2_score(q_sim_i[-(len_valid_i - len_train):], q_obs_i[-(len_valid_i - len_train):])
                BIAS_ii = q_sim_i[-(len_valid_i - len_train):].sum() / q_obs_i[-(len_valid_i - len_train):].sum()
                CC_ii = np.corrcoef([q_sim_i[-(len_valid_i - len_train):], q_obs_i[-(len_valid_i - len_train):]], rowvar=True)[0, 1]
                mean_s_ii = q_sim_i[-(len_valid_i - len_train):].mean()
                mean_o_ii = q_obs_i[-(len_valid_i - len_train):].mean()
                std_s_ii = q_sim_i[-(len_valid_i - len_train):].std()
                std_o_ii = q_obs_i[-(len_valid_i - len_train):].std()
                KGE_ii = 1 - ((CC_ii - 1) ** 2 + (std_s_ii / std_o_ii - 1) ** 2 + (mean_s_ii / mean_o_ii - 1) ** 2) ** 0.5
        
                # Update the results_catchment DataFrame with evaluation metrics for each catchment
                results_catchment.at[i * len(code_list) + ii, 'code'] = code_list.GAGE_ID[ii]
                results_catchment.at[i * len(code_list) + ii, 'lag'] = lag - 1
                results_catchment.at[i * len(code_list) + ii, 'batch'] = batch_size
                results_catchment.at[i * len(code_list) + ii, 'cell'] = state_size
                results_catchment.at[i * len(code_list) + ii, 'RMSE'] = RMSE_ii
                results_catchment.at[i * len(code_list) + ii, 'R2'] = R2_ii
                results_catchment.at[i * len(code_list) + ii, 'std_ratio'] = std_s_ii / std_o_ii
                results_catchment.at[i * len(code_list) + ii, 'CC'] = CC_ii
                results_catchment.at[i * len(code_list) + ii, 'Bias_ratio'] = BIAS_ii
                results_catchment.at[i * len(code_list) + ii, 'KGE_testing'] = KGE_ii



            # Generate date ranges and expand code list for predictions DataFrame
            dates = pd.date_range(start=ini_training, end='2008/12/31')
            list_dates = [dates.date] * len(code_list)
            concatenated_dates = []
            for idx in list_dates:
                concatenated_dates.extend(idx.tolist())

            
            expanded_list = []
            for item in code_list.GAGE_ID.values:
                expanded_list.extend([item] * l_period)
            
            ## Ensure the column is of datetime type before assignment
            #predictions[3] = pd.to_datetime(predictions[3])
            #state_results[2] = pd.to_datetime(state_results[2])
            
            
            # Update predictions DataFrame with observations, simulations, dates, and catchment IDs
            predictions.loc[0, 1] = lag - 1
            predictions.loc[1, 1] = state_size
            predictions.loc[1, 2] = 'ID'
            predictions.loc[1, 0] = 'obs'
            predictions.loc[1, 3] = 'date'
            predictions.loc[0, 3] = ''
            predictions.loc[2:len(expanded_list) + 2, 2] = expanded_list
            predictions.loc[2:len(q_obs) + 2, 0] = q_obs
            predictions.loc[2:len(q_sim) + 2, 1] = q_sim
            predictions.loc[2:len(concatenated_dates) + 2, 3] = concatenated_dates
            

            
            # Update state_results DataFrame with state values, catchment IDs, and dates if state_size equals 1
            if state_size == 1:
                state_results.loc[0, 0] = lag - 1
                state_results.loc[1, 0] = state_size
                state_results.loc[1:, 1] = 'ID'
                state_results.loc[2:len(state_value) + 2, 0] = state_value
                state_results.loc[2:len(expanded_list) + 2, 1] = expanded_list
                state_results.loc[2:len(concatenated_dates) + 2, 2] = concatenated_dates
            
            # Print the results_catchment DataFrame
            print(results_catchment)

           
        else:

            q_sim = q_sim*(ds.y_max - ds.y_min) + ds.y_mean
            q_obs = q_obs*(ds.y_max - ds.y_min) + ds.y_mean

            q_sim = q_sim.flatten()
            q_obs = q_obs.flatten()
            if state_size == 1:
                state_value = state_value.flatten()


            end_train = ds_valid.num_samples
            
            RMSE = mean_squared_error(q_sim[end_train:], q_obs[end_train:])**0.5
            MAE = mean_absolute_error(q_sim[end_train:], q_obs[end_train:])
            R2 = r2_score(q_sim[end_train:], q_obs[end_train:])
            BIAS = q_sim[end_train:].sum() / q_obs[end_train:].sum()
            CC = np.corrcoef([q_sim[end_train:], q_obs[end_train:]],rowvar=True)
            CC = CC[0,1]
            mean_s = q_sim[end_train:].mean()
            mean_o = q_obs[end_train:].mean()
            std_s = q_sim[end_train:].std()
            std_o = q_obs[end_train:].std()
            KGE = 1 - ((CC - 1) ** 2 + (std_s / std_o - 1) ** 2 + (mean_s / mean_o - 1) ** 2) ** 0.5

            results.at[i,'lag'] = lag-1
            results.at[i,'batch'] = batch_size
            results.at[i,'cell'] = state_size
            results.at[i,'RMSE'] = RMSE
            results.at[i,'MAE'] = MAE
            results.at[i,'R2'] = R2
            results.at[i,'CC'] = CC
            results.at[i,'std_ratio'] = std_s / std_o
            results.at[i,'Bias_ratio'] = mean_s / mean_o
            results.at[i,'KGE_testing'] = KGE
            print(results)
            
            # Generate date ranges and expand code list for predictions DataFrame
            dates = pd.date_range(start=ini_training, end='2008/12/31')
            
            # # Ensure the column is of datetime type before assignment
            # state_results[0] = pd.NaT
            # predictions[0] = pd.NaT

            # Convert the entire DataFrame to object dtype
            predictions = predictions.astype(object)
            state_results = state_results.astype(object)

            predictions.iloc[2:,0] = dates.date
            predictions.loc[1,0] = 'date'
            predictions.loc[0,i+2] = lag-1
            predictions.loc[1,i+2] = state_size
            predictions.loc[2:,i+2] = q_sim
            predictions.loc[2:,1] = q_obs
            predictions.loc[1,1] = 'obs'
            
            if state_size == 1:
                state_results.loc[2:,0] = dates.date
                state_results.loc[1,0] = 'date'
                state_results.loc[0,i+1] = lag-1
                state_results.loc[1,i+1] = state_size
                state_results.loc[2:,i+1] = state_value
                
            #print(predictions)
            #print(state_results)

        # Increment the index variable i
        i = i + 1
        
        # Append the trained model to the model_summary list
        model_summary.append(model)



#%%  
# saving files   

# If the model option is "HYDRO", save results, trained models, predictions, and state results
if cfg["model"] == "HYDRO":
    # Define file names based on code, state size, and lag
    name_file = str(cfg["code"]) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_summary_testing.csv'
    # Save results to a CSV file
    results.to_csv(name_file)
    
    name_file = str(cfg["code"]) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_predictions_testing.csv'
    # Convert predictions to a DataFrame and save to a CSV file
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(name_file)

    # If the state size is 1, save state results to a CSV file
    if state_size == 1:
        name_file = str(cfg["code"]) + '_C' + str(state_size) + '_L' + str(lag-1) + '_hydro_state_testing.csv'
        state_results = pd.DataFrame(state_results)
        state_results.to_csv(name_file)

        
# If the model option is not "HYDRO", save results, trained models, predictions, and state results
else:
    # Define file names based on code, state size, and lag
    name_file = str(cfg["code"]) + '_C' + str(state_size) + '_L' + str(lag-1) + '_' + 'regionalhydro_summary_testing.csv'
    # Save results to a CSV file
    results.to_csv(name_file)
    

    name_file = str(cfg["code"]) + '_C' + str(state_size) + '_L' + str(lag-1) + '_' + 'regionalhydro_predictions_testing.csv'
    # Convert predictions to a DataFrame and save to a CSV file
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(name_file)   
    
    name_file = str(cfg["code"]) + '_C' + str(state_size) + '_L' + str(lag-1) + '_regionalhydro_summary_per_catchment_testing.csv'
    results_catchment.to_csv(name_file)
        
    # If the state size is 1, save state results to a CSV file
    if state_size == 1:
        name_file = str(cfg["code"]) + '_C' + str(state_size) + '_L' + str(lag-1) + '_' + 'regionalhydro_state_testing.csv'
        state_results = pd.DataFrame(state_results)
        state_results.to_csv(name_file) 
        





print('--------------------------------------------------------------------------------------------------------------')
print('Testing results are ready')
print('--------------------------------------------------------------------------------------------------------------')


