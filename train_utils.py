'''
This module containes the contains the codes that I wrote for my network trainings.
'''

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from copy import deepcopy
import random
import numpy as np

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def update (model, opt, loss_func, xb, yb):
    '''
    function to do update model in one batch calculation
    
    model     : model that we want to update its weights in one epoch 
    opt       : optimizer
    loss_func : corresponding loss function
    xb, yb    : mini-batch and its labels
    '''
    
    model.train()
    opt.zero_grad()
    
    preds = model(xb)
    loss  = loss_func(preds, yb)
    acc   = accuracy(preds, yb)
    loss.backward()
    opt.step()
    
    return loss.item(), acc.item()


class EarlyStopping():
    def __init__(self, state, patience=10, attribute='loss'):
        '''
        a class for doining early stopping during training
        
        state     : use ES or not (boolean)
        patience  : if we see this number of results after our best result we break the training loop (int)
        attribute : the attribute for validation data that we decide the stopping based on that ('loss' or 'acc')
        '''
        
        self.state     = state
        self.patience  = patience
        self.attribute = attribute
        self.b_model   = nn.Module()                                # best model that is found during trainin
        self.b_opt     = None                                       # optimizer of best model
        self.atr_value = float('inf') if attribute == 'loss' else 0 # valid loss/acc of best model
        self.counter   = 0                                          # if counter==patience then stop training
        

    
def train (
    model, opt,
    train_dl, test_dl,
    epochs,
    loss_func = F.nll_loss, period = 1,
    er_stop = EarlyStopping(state=False)
    ):
    
    '''
    model         : the neural network that we want to train
    opt           : optimizer that we want to use for updating weights
    train_dl      : training data loader
    test_dl       : validation data loader
    epochs        : number of training epochs (int)
    loss_func     : loss function that is used for updating weights
    period        : period for printing training and validation logs (int)
    er_stop       : EarlyStopping object (default is training loop without early-stopping)
    '''
    history = {'train_loss' : [],
               'train_acc'  : [],
               'valid_loss' : [],
               'valid_acc'  : []}
    
    if next(model.parameters()).is_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
       
    ##############################################
    for ep in range(1, epochs+1):
        if ep % period == 0 or ep == 1:
            print (f'\n*** Epoch: {ep} ***')
        
        tmp_loss, tmp_acc = [], []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss, acc = update(model, opt, loss_func, xb, yb)
            tmp_loss.append(loss)
            tmp_acc. append(acc)
        
        history['train_loss'].append(sum(tmp_loss)/len(tmp_loss))
        history['train_acc' ].append(sum(tmp_acc)/len(tmp_acc))
        
        model.eval()
        with torch.no_grad():
            tmp_loss, tmp_acc = [], []
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss  = loss_func(preds, yb)
                acc   = accuracy(preds, yb)
                tmp_loss.append(loss.item())
                tmp_acc. append(acc.item())
            
            history['valid_loss'].append(sum(tmp_loss)/len(tmp_loss))
            history['valid_acc' ].append(sum(tmp_acc)/len(tmp_acc))
            

        if ep % period == 0 or ep == 1:
            print('Train Loss: {:.4f} --- Train Acc {:.2f}\nValid Loss: {:.4f} --- Valid Acc: {:.2f}'.format(
                history['train_loss'][-1], history['train_acc'][-1]*100,
                history['valid_loss'][-1], history['valid_acc'][-1]*100,
            ))
        
        if er_stop.state:
            if er_stop.attribute == 'loss':
                if history['valid_loss'][-1] < er_stop.atr_value:
                    er_stop.atr_value = history['valid_loss'][-1]
                    er_stop.b_model_state   = deepcopy(model).state_dict()
                    er_stop.b_opt_state     = deepcopy(opt).state_dict()
                    er_stop.counter   = 0 
                else:
                    er_stop.counter  += 1
                    
            elif er_stop.attribute == 'acc':
                if history['valid_acc'][-1] > er_stop.atr_value:
                    er_stop.atr_value = history['valid_acc'][-1]
                    er_stop.b_model_state   = deepcopy(model).state_dict()
                    er_stop.b_opt_state     = deepcopy(opt).state_dict()
                    er_stop.counter   = 0
                else:
                    er_stop.counter  += 1
            
            else:
                print('The attribute should be either <loss> or <acc>')
                break

            if er_stop.counter == er_stop.patience:
                break

        # No early stopping, but we still use er_stop.attribute to save the best model
        else:
            if er_stop.attribute == 'loss':
                if history['valid_loss'][-1] < er_stop.atr_value:
                    er_stop.atr_value = history['valid_loss'][-1]
                    er_stop.b_model_state   = deepcopy(model).state_dict()
                    er_stop.b_opt_state     = deepcopy(opt).state_dict()
  
                    
            elif er_stop.attribute == 'acc':
                if history['valid_acc'][-1] > er_stop.atr_value:
                    er_stop.atr_value = history['valid_acc'][-1]
                    er_stop.b_model_state   = deepcopy(model).state_dict()
                    er_stop.b_opt_state     = deepcopy(opt).state_dict()
   

    checkpoint = {
            'model': er_stop.b_model_state,
            'optimizer': er_stop.b_opt_state
            }

    return history, checkpoint


