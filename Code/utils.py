# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:14:04 2021
modified Jan 14
@author: Luis De la Fuente
"""

from importing import *
from datetime import timedelta
from torch.utils.data import DataLoader, SequentialSampler
import torch
from tqdm import tqdm
import sys
import random 

#%%

def creating_dataset(code, code_list, code_att,n_attributes, lag, ini_training, training_last_day, validation_last_day):
    """ """


    x_max = [] # try to incorporate in default values of the fx
    x_min = []
    x_mean = []
    y_max = []
    y_min = []
    y_mean = []
    
    # Initialize a DataFrame for scaling information for the target variable
    if not code_list.empty:
        y_scaler = pd.DataFrame(np.zeros([len(code_list), 3]), columns=['y_max', 'y_min', 'y_mean'], index=code_list.GAGE_ID)
    else:
        y_scaler = pd.DataFrame()



    if code == 1000000:

                
        x_max_i = []
        x_min_i = []
        x_mean_i = []
        y_max_i = []
        y_min_i = []
        y_mean_i = []

        random.shuffle(code_list.GAGE_ID)
        z=1
        print('Preparing the dataset')
        for code_i in code_list.GAGE_ID:
            
            PP_i, PET_i, Q_i = load_data(code_i, lag) 

            
            if Q_i.isna().any().any():
                # Print a message indicating the current code has NaN values and increment the counter
                print('code ', code_i, 'has nan values')
                print('#',z," ",code_i)
                z = z + 1

            else:

                # Split the data into training, testing, and validation sets
                PP_train_i = PP_i[PP_i.index <= training_last_day]
                PET_train_i = PET_i[PET_i.index <= training_last_day]
                Q_train_i = Q_i[Q_i.index <= training_last_day]
                
                # Copy the entire dataset for testing
                PP_test_i = PP_i.copy()
                PET_test_i = PET_i.copy()
                Q_test_i = Q_i.copy()
                
                # Split the data into validation sets
                PP_valid_i = PP_i[PP_i.index <= validation_last_day]
                PET_valid_i = PET_i[PET_i.index <= validation_last_day]
                Q_valid_i = Q_i[Q_i.index <= validation_last_day]


                #print('Q_i.Q_obs.min():',Q_i.Q_obs.min())

                if z==1:
                  
                    # Copy training, validation, and testing data for PP, PET, and Q
                    PP_train = PP_train_i.copy()
                    PET_train = PET_train_i.copy()
                    Q_train = Q_train_i.copy()
                    
                    PP_test = PP_test_i.copy()
                    PET_test = PET_test_i.copy()
                    Q_test = Q_test_i.copy()
                    
                    PP_valid = PP_valid_i.copy()
                    PET_valid = PET_valid_i.copy()
                    Q_valid = Q_valid_i.copy()
                    
                    # Create a torch dataset for training data and store normalization statistics
                    ds = torch_dataset(PP_train, PET_train, Q_train, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=True)
                    x_max_i = ds.x_max
                    x_min_i = ds.x_min
                    x_mean_i = ds.x_mean
                    y_max_i = ds.y_max
                    y_min_i = ds.y_min
                    y_mean_i = ds.y_mean
                    
                    # Update y_scaler DataFrame with normalization statistics for the current code

                    y_scaler.loc[code_i, 'y_min'] = y_min_i
                    y_scaler.loc[code_i,'y_mean'] = y_mean_i
                    y_scaler.loc[code_i,'y_max'] = y_max_i


                    # Create and append attribute tensor to training dataset
                    att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values, code_i), dtype=torch.float32)
                    att_tensor = att_tensor.repeat(ds.x.shape[0], 1)
                    ds.x = torch.cat((ds.x, att_tensor), 1)
                    len_train = len(ds.y)
                    
                    # Create a torch dataset for validation data and append attribute tensor
                    ds_valid = torch_dataset(PP_valid, PET_valid, Q_valid, lag, ini_training, ds.x_max, ds.x_min, ds.x_mean, ds.y_max, ds.y_min, ds.y_mean, istrain=False)
                    att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values, code_i), dtype=torch.float32)
                    att_tensor = att_tensor.repeat(ds_valid.x.shape[0], 1)
                    ds_valid.x = torch.cat((ds_valid.x, att_tensor), 1)
                    len_valid = len(ds_valid.y)
                    
                    # Create a torch dataset for testing data and append attribute tensor
                    ds_full = torch_dataset(PP_test, PET_test, Q_test, lag, ini_training, ds.x_max, ds.x_min, ds.x_mean, ds.y_max, ds.y_min, ds.y_mean, istrain=False)
                    att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values, code_i), dtype=torch.float32)
                    att_tensor = att_tensor.repeat(ds_full.x.shape[0], 1)
                    ds_full.x = torch.cat((ds_full.x, att_tensor), 1)
                    len_test = len(ds_full.y)


                    
                else:
                    
                    # Create a torch dataset for the current training data, add attributes tensor, and update training length
                    ds_i = torch_dataset(PP_train_i, PET_train_i, Q_train_i, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=True)
                    att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values, code_i), dtype=torch.float32)
                    att_tensor = att_tensor.repeat(ds_i.x.shape[0], 1)
                    ds_i.x = torch.cat((ds_i.x, att_tensor), 1)
                    len_train_i = len(ds_i.y)
                    
                    # Update y_scaler DataFrame with normalization statistics for the current code
                    y_scaler.loc[code_i,'y_max'] = ds_i.y_max
                    y_scaler.loc[code_i,'y_min'] = ds_i.y_min
                    y_scaler.loc[code_i,'y_mean'] = ds_i.y_mean
                    
                    # Create a torch dataset for the current validation data and add attributes tensor
                    ds_valid_i = torch_dataset(PP_valid_i, PET_valid_i, Q_valid_i, lag, ini_training, ds_i.x_max, ds_i.x_min, ds_i.x_mean, ds_i.y_max, ds_i.y_min, ds_i.y_mean, istrain=False)
                    att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values, code_i), dtype=torch.float32)
                    att_tensor = att_tensor.repeat(ds_valid_i.x.shape[0], 1)
                    ds_valid_i.x = torch.cat((ds_valid_i.x, att_tensor), 1)
                    len_valid_i = len(ds_valid_i.y)
                    
                    # Create a torch dataset for the current testing data and add attributes tensor
                    ds_full_i = torch_dataset(PP_test_i, PET_test_i, Q_test_i, lag, ini_training, ds_i.x_max, ds_i.x_min, ds_i.x_mean, ds_i.y_max, ds_i.y_min, ds_i.y_mean, istrain=False)
                    att_tensor = torch.tensor(np.append(code_att.loc[code_i][-n_attributes:].values, code_i), dtype=torch.float32)
                    att_tensor = att_tensor.repeat(ds_full_i.x.shape[0], 1)
                    ds_full_i.x = torch.cat((ds_full_i.x, att_tensor), 1)
                    len_test_i = len(ds_full_i.y)
                    #print('ds_full_i.y.numpy().min(): ', ds_full_i.y.numpy().min())
                    
                    # Concatenate current dataset with overall datasets for training, validation, and testing
                    ds.x = torch.cat((ds.x, ds_i.x), 0)
                    ds.y = torch.cat((ds.y, ds_i.y), 0)
                    ds.x = ds.x.float()
                    
                    ds_valid.x = torch.cat((ds_valid.x, ds_valid_i.x), 0)
                    ds_valid.y = torch.cat((ds_valid.y, ds_valid_i.y), 0)
                    ds_valid.x = ds_valid.x.float()
                    
                    ds_full.x = torch.cat((ds_full.x, ds_full_i.x), 0)
                    ds_full.y = torch.cat((ds_full.y, ds_full_i.y), 0)
                    ds_full.x = ds_full.x.float()

                    

                    
                print(z)
                z = z+1
                


        # Compute and assign normalization statistics for input features (x) across datasets
        ds.x_max = ds.x.max(axis=0)
        ds.x_min = ds.x.min(axis=0)
        ds.x_mean = ds.x.mean(axis=0)
        ds_valid.x_max = ds.x_max
        ds_valid.x_min = ds.x_min
        ds_valid.x_mean = ds.x_mean
        ds_full.x_max = ds.x_max
        ds_full.x_min = ds.x_min
        ds_full.x_mean = ds.x_mean
        
        # Compute and assign normalization statistics for target variable (y) across datasets
        ds.y_max = ds.y.max()
        ds.y_min = ds.y.min()
        ds.y_mean = ds.y.mean()
        ds_valid.y_max = ds.y_max
        ds_valid.y_min = ds.y_min
        ds_valid.y_mean = ds.y_mean
        ds_full.y_max = ds.y_max
        ds_full.y_min = ds.y_min
        ds_full.y_mean = ds.y_mean
        
        # Convert normalization statistics to numpy arrays for further processing
        x_max = ds.x_max.values.numpy()
        x_min = ds.x_min.values.numpy()
        x_mean = ds.x_mean.numpy()
        y_max = ds.y_max.numpy()
        y_min = ds.y_min.numpy()
        y_mean = ds.y_mean.numpy()
        
        # Assign the number of samples for training, validation, and full datasets
        ds.num_samples = len(ds.y)
        ds_full.num_samples = len(ds_full.y)
        ds_valid.num_samples = len(ds_valid.y)
                

            
    else:
        
        
        # Load the data for the given code and lag
        PP, PET, Q = load_data(code, lag)
        
        # Split the data into training, testing, and validation sets
        PP_train = PP[PP.index <= training_last_day]
        PET_train = PET[PET.index <= training_last_day]
        Q_train = Q[Q.index <= training_last_day]
        
        PP_test = PP.copy()
        PET_test = PET.copy()
        Q_test = Q.copy()
        
        PP_valid = PP[PP.index <= validation_last_day]
        PET_valid = PET[PET.index <= validation_last_day]
        Q_valid = Q[Q.index <= validation_last_day]
        

        # Create a torch dataset for the training data and store normalization statistics
        ds = torch_dataset(PP_train, PET_train, Q_train, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=True)
        x_max = ds.x_max
        x_min = ds.x_min
        x_mean = ds.x_mean
        y_max = ds.y_max
        y_min = ds.y_min
        y_mean = ds.y_mean
        len_train = len(ds.y)
        print('training:', len_train)
        
        # Create a torch dataset for the validation data and adjust dataset properties
        ds_valid = torch_dataset(PP_valid, PET_valid, Q_valid, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=False)
        #len_valid = len(ds_valid.y)
        #ds_valid.x = ds_valid.x[-(len_valid - len_train):, :]
        #ds_valid.y = ds_valid.y[-(len_valid - len_train):]
        #ds_valid.num_samples = len(ds_valid.y)
        len_valid = len(ds_valid.y)
        print('validation:', len_valid)
        
        # Create a torch dataset for the testing data and prepare data loader
        ds_full = torch_dataset(PP_test, PET_test, Q_test, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain=False)
        len_test = len(ds_full.y) #- len_train - len_valid
        print('testing:', len_test)
        #sampler_test = SequentialSampler(ds_full)
        #loader_test = DataLoader(ds_full, batch_size=batch_size, sampler=sampler_test, shuffle=False)


    return ds, ds_valid, ds_full, y_scaler, len_train, len_valid , len_test       


  
#%%
def load_data(code, warm_up):
    PP, PET, Q = importing(code)
       
    Q['Q_obs'] = pd.to_numeric(Q['Q_obs'],errors = 'coerce')

    Q = Q[:PP.index[-1]]

    Q_nan = Q.dropna()
    if PP.index[0] + pd.DateOffset(days=warm_up) < Q_nan.index[0]:
        PP = PP[Q_nan.index[0] - pd.DateOffset(days=warm_up):]
        PET = PET[Q_nan.index[0] - pd.DateOffset(days=warm_up):]
        Q = Q[Q_nan.index[0]:]
    else:
        Q = Q[PP.index[0] + pd.DateOffset(days=warm_up):]
    return PP, PET, Q

#%%
class torch_dataset():
    
    def __init__(self, PP,PET,Q, lag, ini_training, x_max, x_min, x_mean, y_max, y_min, y_mean, istrain):
              
        for i in range(1,lag):

            PP_name = 'PP_' + str(i)
            PP_copy = PP.copy()
            PP_copy[PP_name] = PP['PP'].shift(i)
            PP = PP_copy.copy()

            
            PET_name = 'PET_' + str(i)
            PET_copy = PET.copy()
            PET_copy[PET_name] = PET['PET'].shift(i)
            PET = PET_copy.copy()
                  
        X = pd.concat([PP, PET], axis=1)
        X = X.drop('basin', axis=1)
        X.at[:,'Q'] = Q.Q_obs #        X['Q'] = Q.Q_obs

        #PP[PP.index <= validation_last_day]
        #print(ini_training)
        X = X[X.index >= ini_training]
        X = X.dropna()
        X = X.drop('Q', axis=1)
        x = X.values
        Q = Q.loc[X.index]
        #print(Q)
        y = Q.Q_obs.values
        
        if istrain:          
            self.x_max = x.max(axis=0)
            self.x_min = x.min(axis=0)
            self.x_mean = x.mean(axis=0) #[-1,1]
        
            self.y_max = y.max()
            self.y_min = y.min()
            self.y_mean = y.mean() #[-1,1]
        else:
            self.x_max = x_max
            self.x_min = x_min
            self.x_mean = x_mean
            self.y_max = y_max
            self.y_min = y_min
            self.y_mean = y_mean
                
        y = (y - self.y_mean)/(self.y_max - self.y_min)
        x = (x - self.x_mean)/(self.x_max - self.x_min) 
        
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.num_samples = self.x.shape[0]       
        
    def __len__(self):
        return self.num_samples   

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


#%%
def train_epoch(model, optimizer, loss_func, loader, epoch, loader_valid, patience, mean_valid_losses,valid_period,  DEVICE):
     

    stopping = False
    
    model.train()
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')
    train_losses = []
    pred_year = []
    state_year = []
    for data in pbar:
        
        optimizer.zero_grad()# delete old gradients
        x, y = data
        x, y = x.to(DEVICE), y.to(DEVICE)
        #y = y.resize(len(y),1) 
        y.resize_(len(y),1) 
        
        model.epoch = epoch
        model.DEVICE = DEVICE
        predictions = model(x)[0]
        c_pred = model.c_t.data
        #at_w = model.at_w.data

        loss = loss_func(predictions, y) 

        loss.backward() 

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step() # perform parameter update
        
        
        
        train_losses.append(loss.item())

        #pbar.set_postfix_str(f"Loss: {loss.item():5f}")
       
    total = sum(train_losses)
    length = len(train_losses)
    mean_loss = total/length
    print(f"Loss_train: {mean_loss:.6f}")
    
    # Iterate in batches over valid set
    valid_losses = []

    for data in loader_valid:

        x_valid, y_valid = data
       
        x_valid, y_valid = x_valid.to(DEVICE), y_valid.to(DEVICE)
        #y_valid = y_valid.resize(len(y_valid),1) 
        y_valid.resize_(len(y_valid),1) 
        
        pred_valid = model(x_valid)[0]
                
        loss_valid = loss_func(pred_valid[-valid_period:], y_valid[-valid_period:])

        valid_losses.append(loss_valid.item())



    total_valid = sum(valid_losses)
    length_valid = len(valid_losses)
    epoch_valid_loss = total_valid/length_valid
    mean_valid_losses.append(epoch_valid_loss)
    
    print(f"Loss_valid: {epoch_valid_loss:.6f}")
    
   

    if epoch >= patience:
        if (mean_valid_losses[epoch-1] > mean_valid_losses[epoch - patience]):
            print("Early stopping")
            stopping = True

    return stopping, mean_valid_losses


def train_epoch_RF(model, optimizer, loss_func, loader, epoch, loader_valid, patience, mean_valid_losses, df_RF, df_regression, y_scaler, valid_period, DEVICE):
     

    stopping = False
    
    model.train()
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')
    train_losses = []
    pred_year = []
    state_year = []
    for data in pbar:
        
        optimizer.zero_grad()# delete old gradients
        x, y = data
        x, y = x.to(DEVICE), y.to(DEVICE)
        #y = y.resize(len(y),1) #y.resize_(len(y),1) #
        y.resize_(len(y),1) #y.resize_(len(y),1) #
        
        model.epoch = epoch
        model.DEVICE = DEVICE
        predictions = model(x,df_RF,df_regression)[0]
        c_pred = model.c_t.data
     
        loss = loss_func(predictions, y) 

        loss.backward() 

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step() # perform parameter update
        
        train_losses.append(loss.item())
        
        if model.hydro_lstm_regional.state_size == 1:
            catch_ID =  int(np.round(x[0,-1].detach().numpy(),decimals=0))

            df_RF.iloc[df_RF.index.get_loc(catch_ID), :int(513 * 8)] = model.hydro_lstm_regional.weight_input.data.detach().numpy().flatten()
            df_RF.iloc[df_RF.index.get_loc(catch_ID), int(513 * 8):int(513 * 8 + 4)] = model.hydro_lstm_regional.weight_recur.data.detach().numpy().flatten()
            df_RF.iloc[df_RF.index.get_loc(catch_ID),int(513 * 8 + 4):int(513 * 8 + 8)] = model.hydro_lstm_regional.bias.data.detach().numpy().flatten()
            
            a, b = model.regression.parameters()
            a = a.detach().numpy().flatten()
            b = b.detach().numpy().flatten()            
            df_regression.at[catch_ID,0] = a
            df_regression.at[catch_ID,1] = b
       
    total = sum(train_losses)
    length = len(train_losses)
    mean_loss = total/length
    print(f"Loss_train: {mean_loss:.6f}")
    
    # Iterate in batches over valid set
    valid_losses = []

    for data in loader_valid:

        catch_ID =  int(np.round(x[0,-1].detach().numpy(),decimals=0)) 
        x_valid, y_valid = data
       
        x_valid, y_valid = x_valid.to(DEVICE), y_valid.to(DEVICE)
        #y_valid = y_valid.resize(len(y_valid),1) 
        y_valid.resize_(len(y_valid),1)
        
        pred_valid = model(x_valid, df_RF, df_regression)[0]
         
        q_sim_i = (pred_valid[-valid_period:]*(y_scaler.loc[catch_ID]['y_max'] - y_scaler.loc[catch_ID]['y_min']) + y_scaler.loc[catch_ID]['y_mean']).detach().numpy().flatten()
        q_obs_i = (y_valid[-valid_period:]*(y_scaler.loc[catch_ID]['y_max'] - y_scaler.loc[catch_ID]['y_min']) + y_scaler.loc[catch_ID]['y_mean']).detach().numpy().flatten()

        CC_ii = np.corrcoef([q_sim_i, q_obs_i],rowvar=True)
        CC_ii = CC_ii[0,1]
        mean_s_ii = q_sim_i.mean()
        mean_o_ii = q_obs_i.mean()
        std_s_ii = q_sim_i.std()
        std_o_ii = q_obs_i.std()  
        KGE_ii = 1 - ((CC_ii - 1) ** 2 + (std_s_ii / std_o_ii - 1) ** 2 + (mean_s_ii / mean_o_ii - 1) ** 2) ** 0.5
        
                
        loss_valid = KGE_ii
        
        valid_losses.append(loss_valid.item())
        


        


    total_valid = sum(valid_losses)
    length_valid = len(valid_losses)
    epoch_valid_loss = np.median(valid_losses)
    
    mean_valid_losses.append(epoch_valid_loss)
    
    print(f"KGE_valid: {epoch_valid_loss:.6f}")
    

    if epoch >= patience:
        if (mean_valid_losses[epoch-1] > mean_valid_losses[epoch - patience]):# and (mean_loss < epoch_valid_loss):
            print("Early stopping")
            stopping = True

    return stopping, mean_valid_losses, df_RF, df_regression
    
def position_highest(lst):
    if not lst:
        return None  # Return None for an empty list

    max_value = max(lst)
    position = [index for index, value in enumerate(lst) if value == max_value]

    return position[0]

