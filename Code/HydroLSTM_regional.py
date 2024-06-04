# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:18:48 2021

@author: Newpc
"""
import torch
import torch.nn as nn
import numpy as np

 
#%%

class Model_hydro_lstm_regional(nn.Module):
    def __init__(self, input_size: int, lag: int, state_size: int, dropout: float, attributes: int ):
        super(Model_hydro_lstm_regional, self).__init__()
        self.input_size = input_size #??input_size*lag
        self.state_size = state_size
        self.dropout = nn.Dropout(p=dropout)
        self.attributes_size = attributes
        self.h_t = torch.zeros(1)  
        self.c_t = torch.zeros(1)  
        self.hydro_lstm_regional = HYDROLSTM_regional(input_size, lag, state_size, attributes_size = attributes)
        self.regression = nn.Linear(state_size, 1, bias=True)


    def forward(self, x, RF_weights, RF_regression):

        #Initialization of the states 
        h_0 = torch.ones(1, self.state_size).to(
            self.DEVICE)*self.h_t.data[-1].to(self.DEVICE)
        c_0 = torch.ones(1, self.state_size).to(
            self.DEVICE)*self.c_t.data[-1].to(self.DEVICE)
        

        catch_ID =  int(np.round(x[0,-1].detach().numpy(),decimals=0))
        self.regression.weight.data.copy_(torch.tensor(RF_regression.loc[catch_ID][0], requires_grad=True).float())
        self.regression.bias.data.copy_(torch.tensor(RF_regression.loc[catch_ID][1], requires_grad=True).float())
        
        self.h_t, self.c_t = self.hydro_lstm_regional(x, h_0, c_0, RF_weights)


        if self.state_size != 1:
            self.h_t_dropout = self.dropout(self.h_t)
            q_t = self.regression(self.h_t_dropout)

        else:
            q_t = self.regression(self.h_t)

        q_t = q_t[1:]
        return q_t, self.h_t, self.c_t

    
#%%

class HYDROLSTM_regional(nn.Module):
    def __init__(self, input_size: int, lag:int, state_size: int, attributes_size: int):
        super(HYDROLSTM_regional, self).__init__()
        self.input_size = input_size
        self.lag = lag
        self.state_size = state_size
        self.attributes_size = attributes_size
        self.weight_input = []
        self.bias = []
        self.catch_ID =[]

    
        
        # create tensors parameters    

        if self.state_size == 1:
            self.weight_input = nn.Parameter(torch.FloatTensor(4, input_size))  # (#input x 4 * #state variables)
        else:
            self.weight_input = nn.Parameter(torch.FloatTensor(state_size, 4, input_size))  # (#input x 4 * #state variables)
            


        self.weight_recur = nn.Parameter(torch.FloatTensor(4, state_size))
        self.bias = nn.Parameter(torch.FloatTensor(4, state_size ))

        # Initialize parameters for LSTM
        nn.init.xavier_uniform_(self.weight_input.data, gain=1.0)
        nn.init.xavier_uniform_(self.weight_recur.data, gain=1.0)
        nn.init.xavier_uniform_(self.bias.data, gain=1.0)


    def forward(self, x: torch.Tensor, h_0, c_0, RF_weights):

        batch_size = x.size(0)  # checking the batch size

        x_transp = torch.transpose(x[:,:-self.attributes_size -1], 0, 1)
        

        if self.catch_ID != x[0,-1]:
            if self.state_size == 1:
                       
                catch_ID =  int(np.round(x[0,-1].detach().numpy(),decimals=0))
                self.weight_input.data = torch.tensor((RF_weights.loc[catch_ID][:int(self.lag * 8)]).values.reshape(4,self.input_size), requires_grad=True).float()  
                self.weight_recur.data = torch.tensor((RF_weights.loc[catch_ID][int(self.lag * 8): int(self.lag * 8 + 4)]).values.reshape((4, 1)), requires_grad=True).float()  
                self.bias.data = torch.tensor((RF_weights.loc[catch_ID][int(self.lag * 8 + 4): int(self.lag * 8 + 8)]).values.reshape((4, 1)), requires_grad=True).float()

            else:
                catch_ID =  int(np.round(x[0,-1].detach().numpy(),decimals=0))
                self.weight_input.data[0, 4, input_size] = torch.tensor((RF_weights.loc[catch_ID][:int(self.lag * 8)]).values.reshape(4,self.input_size), requires_grad=True).float()     
                
            self.catch_ID = x[0,-1]
            

        h_n = h_0
        c_n = c_0

        h_0 = torch.diagflat(h_0)
        c_0 = torch.diagflat(c_0)


        for it in range(batch_size):
            x_t = x_transp.data[:, it]
            #x_t = x_t.resize(self.input_size, 1)
            x_t.resize_(self.input_size, 1)
        
            if self.state_size == 1:
                gates = (torch.addmm(self.bias, self.weight_recur, h_0) + torch.mm(self.weight_input, x_t))  # calculate gates

            else:
                x_t = x_t.unsqueeze(dim=0)
                x_t = x_t.repeat(self.state_size, 1, 1)
                para =self.weight_input
                
                sum1 = torch.bmm(self.weight_input, x_t)
                sum1 = sum1.squeeze()
                sum1 = torch.transpose(sum1, 0, 1)

                # calculate gates
                gates = (torch.addmm(self.bias, self.weight_recur, h_0) + sum1)
            f, i, o, g = gates.chunk(4, dim=0)
            f = torch.sigmoid(f)  # calculate activation [0,1]
            i = torch.sigmoid(i)  # calculate activation [0,1]
            o = torch.sigmoid(o)  # calculate activation [0,1]
            g = torch.tanh(g)  # calculate activation [-1,1]

            if self.state_size == 1:
                c_1 = f * c_0 + i * g

            else:
                c_1 = torch.mm(f, c_0) + torch.mul(i, g)
                c_1 = torch.diagflat(c_1)


            h_1 = torch.mul(o, torch.tanh(c_1))  
            
            h_0 = h_1
            c_0 = c_1

            h_1 = torch.diag(h_1, 0)
            c_1 = torch.diag(c_1, 0)
            h_1 = h_1.unsqueeze(dim=0)
            c_1 = c_1.unsqueeze(dim=0)
            h_n = torch.cat((h_n, h_1), 0)
            c_n = torch.cat((c_n, c_1), 0)


        return h_n, c_n
    
