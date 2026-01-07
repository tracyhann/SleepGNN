import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils
from src.dataset import ParticipantFCDataset, construct_participant_fc_dataset
from src.model import SleepGNN
from src.train import spatial_train
from src.evaluate import spatial_evaluate
from scripts.configs import MODELS
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GATv2Conv
import numpy as np
import torch.nn as nn

import pandas as pd
import random
import torch
from data.splits.sub_splits import TRAIN_SUBS, TEST_SUBS, TEST_SUBS_ALL_STAGES, TEST_SUBS_WAKE_ONLY, TEST_SUBS_STAGE_BALANCED, TRAIN_SUBS_STAGE_BALANCED
from scripts.train_variants_random import train_loader, train_loader_deduplicated, test_loader, test_loader_group1, test_loader_group2

if torch.cuda.is_available():
    print("✅ Cuda is available")
    device = torch.device("cuda")
#if torch.backends.mps.is_available():
    #print("✅ MPS is available")
    #device = torch.device("mps")
else:
    print("❌ GPU is not available")
    device = torch.device("cpu")
print(f'Using device: {device}')

# determined from 5 fold cv in 16 combos, see outputs/logs/cv
BEST_COMBO = (0., 0.001, 0.0001)  #0., 0.001, 0.0001
EPOCHS = 30

folder = '1118_294_bs4_con0.1'

drop, lr, wd = BEST_COMBO
utils.set_seed(34)

variants = list(MODELS.keys()) # model variant names (keys to config)
print(variants[:])
# patience = 5
for variant in variants[1:]:
    model = SleepGNN(
             roi_conv_type=SAGEConv,
             net_conv_type=MODELS[variant]['attention'], 
             num_roi_layers=2,
             num_network_layers=2,
             input_dim=294,
             hidden_dim=64,
             num_nodes=294,
             num_network=19-4,
             output_num_classes=4,
             num_heads=1,
             edge_filter=None, # use both positive and negative edges for message passing
             edge_retaintion=[0.999, 1], # generate a random subset of edges for Network-level attention communication
             activation= MODELS[variant]['activation'], 
             dropout = drop,
             device = device
        )
    
    print(model)

    model, best_model, best_epoch, logs = spatial_train(model, train_loader, val_loader = test_loader,
                                                 num_epoch = EPOCHS, patience=10, lr = lr, weight_decay=wd, 
                                                 smoothing = 0., con_weight=0.1, verbose = True, device= device)
    print('Best epoch: ', best_epoch)

    best_report_train = spatial_evaluate(best_model, model_name = f'{variant}_best model on train set', 
                                           loader = train_loader_deduplicated, 
                                           verbose = False, 
                                           plt_confusion_matrix = True, 
                                           show_plot= False, save_plot_path=f'outputs/plots/cm/{folder}/{variant}_no_aug',device=device)
    best_report_test = spatial_evaluate(best_model, model_name = f'{variant}_best model on test set', 
                                           loader = test_loader, 
                                           verbose = False, 
                                           plt_confusion_matrix = True, 
                                           show_plot= False, save_plot_path=f'outputs/plots/cm/{folder}/{variant}_no_aug',device=device)
   
    logs['best_epoch'] = best_epoch
    logs['best_train'] = best_report_train
    logs['best_test'] = best_report_test

    final_report_train = spatial_evaluate(model, model_name = f'{variant}_final model on train set', 
                                           loader = train_loader_deduplicated, 
                                           verbose = False, 
                                           plt_confusion_matrix = True, 
                                           show_plot= False, save_plot_path=f'outputs/plots/cm/{folder}/{variant}_no_aug',device=device)
    final_report_test = spatial_evaluate(model, model_name = f'{variant}_final model on test set', 
                                           loader = test_loader, 
                                           verbose = False, 
                                           plt_confusion_matrix = True, 
                                           show_plot= False, save_plot_path=f'outputs/plots/cm/{folder}/{variant}_no_aug',device=device)
    '''
    best_report_group1 = spatial_evaluate(best_model, model_name = f'{variant}_best model on test group 1', 
                                           loader = test_loader_group1, 
                                           verbose = False, 
                                           plt_confusion_matrix = True, 
                                           show_plot= False, save_plot_path=f'outputs/plots/cm/{folder}/{variant}_no_aug')
    best_report_group2 = spatial_evaluate(best_model, model_name = f'{variant}_best model on test group 2', 
                                           loader = test_loader_group2, 
                                           verbose = False, 
                                           plt_confusion_matrix = True, 
                                           show_plot= False, save_plot_path=f'outputs/plots/cm/{folder}/{variant}_no_aug')
    '''
    logs['final_train'] = final_report_train
    logs['final_test'] = final_report_test


    utils.save_train_log(logs, f'{variant}_no_aug_{folder}', root=f"outputs/logs/train/{folder}") #f'{variant}_no_aug_1xrownorm'
    utils.save_trained_model(best_model, variant, variant_name= f'{variant}_BEST_294_2avg', root=f'models/{folder}')
    utils.save_trained_model(model, variant, variant_name= f'{variant}_FINAL_294_2avg', root=f'models/{folder}') 


import matplotlib.pyplot as plt                                                                        
plt.close('all')