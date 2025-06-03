import sys
import matplotlib.pylab as plt
import torch
from tqdm import tqdm
import os
import gpytorch
import numpy as np
from numpy.random import RandomState
import random
sys.path.append('../')
from dataset import load_dataset
from models.gp_rff_dp import RFF_GPLVM
from visualizer import Visualizer
from metrics import (knn_classify,
                     mean_squared_error,
                     r_squared)
import time

def save_models(model, optimizer, epoch, losses, result_dir, data_name, save_model=True):
    '''

    Parameters
    ----------
    model
    optimizer
    epoch
    losses
    result_dir  :           result saving path
    data_name   :           data name
    jj          :           number of experiment repetition
    save_model  :           indication if to saving model

    Returns
    -------

    '''
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch,
             'losses': losses}
    if save_model:
        log_dir = result_dir + f"{data_name}_epoch{epoch}.pt"
        torch.save(state, log_dir)


random_seed = 0

device = 'cpu'
datasets = 's-curve-torch'


""" ##-------------------------------- data loader ------------------------------------------------"""
# Load Dataset
rng = RandomState(random_seed)
ds = load_dataset(rng, datasets, 'gaussian')
Y = ds.Y / 255.

""" ##-------------------------------- Parameters Settings ------------------------------------------------"""
setting_dict = {}
setting_dict['num_m'] = 2              # if num_m = 1, it is using SE kernel
setting_dict['num_sample_pt'] = 25
setting_dict['num_total_pt'] = setting_dict['num_m'] * setting_dict['num_sample_pt']
setting_dict['num_batch'] = 1
setting_dict['lr_hyp'] = .01
setting_dict['iter'] = 6200
setting_dict['num_repexp'] = 1
setting_dict['kl_option'] = True  # if adding X regularization in loss function
setting_dict['noise_err'] = 100.0
setting_dict['latent_dim'] = ds.latent_dim
setting_dict['N'] = ds.Y.shape[0]
setting_dict['M'] = ds.Y.shape[1]

if setting_dict['num_m'] ==1:
    model_name = f"RFF_GPLVM_SE_{setting_dict['num_sample_pt']}"
else:
    model_name = f"RFF_GPLVM_SM_{setting_dict['num_m']}_{setting_dict['num_sample_pt']}"
res_dir = f'/root/scalable_rflvm/figures/{model_name}/'
viz = Visualizer(res_dir+'figures', ds)

GPLVM_model = RFF_GPLVM(setting_dict['num_batch'],
                        setting_dict['num_sample_pt'],
                        setting_dict,
                        Y,
                        device=device).to(device)

optimizer = torch.optim.Adam([p for name, p in GPLVM_model.named_parameters() if name != 'Z_phi'], lr=setting_dict['lr_hyp'])

optimizer_zphi = torch.optim.Adam([GPLVM_model.Z_phi], lr=0.001) 

epochs_iter = tqdm(range(setting_dict['iter']+1), desc="Epoch")

start = time.time()

for i in epochs_iter:

    # Update likelihood block 
    optimizer.zero_grad()
    losstotal = GPLVM_model.compute_loss(batch_y = Y, kl_option=setting_dict['kl_option'])
    losstotal.backward()
    optimizer.step()
    
    # update z block 
    optimizer_zphi.zero_grad()
    loss_zphi = GPLVM_model.compute_zphi_loss(batch_y = Y)
    loss_zphi.backward()
    optimizer_zphi.step()
        
    # other blocks 
    GPLVM_model.inference_v()
    GPLVM_model.inference_alpha()

    if i%200==0:
        print(f'\nELBO: {losstotal.item()}')
        print(f"X_KL: {GPLVM_model._kl_div_qp().item()}")
    
        if ds.is_categorical:
            knn_acc = knn_classify(GPLVM_model.mu_x.cpu().detach().numpy(), ds.labels, rng)
            print('KNN acc', knn_acc)
            np.save(f'/root/scalable_rflvm/X_data/{datasets}_X.npy', GPLVM_model.mu_x.cpu().detach().numpy())
        else: 
            viz.plot_iteration(i + 1,  Y=0,  F=0,  K=0, X=GPLVM_model.mu_x.cpu().detach().numpy())
            
        end = time.time()
        times = end - start 
        print(f'Times: {times}')

        if ds.has_true_X:
            r2_X = r_squared(GPLVM_model.mu_x.cpu().detach().numpy(), ds.X)
            print(f'R2 X: {r2_X}')

        print("\n")