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

model_folder = 'models/1118_294_bs4_con0.1'
folder = '1118_294_bs4_con0.1'

drop, lr, wd = BEST_COMBO
utils.set_seed(34)

variants = list(MODELS.keys()) # model variant names (keys to config)
print(variants[:])
variants = ['baseline', 'sigmoid+gelu', 'tanh+gelu']
# patience = 5
for variant in variants[:]:
    model_dir = os.path.join(model_folder, variant, variant+'_BEST_294_2avg.pt')
    best_model = torch.load(model_dir, map_location=device, weights_only=False)
    
    #print(best_model)
    print('\n===== Evaluating: ', variant, ' ===== \n')

    best_report_group1 = spatial_evaluate(best_model, model_name = f'{variant}_best model on test group 1', 
                                           loader = test_loader_group1, device=device,
                                           verbose = True, 
                                           plt_confusion_matrix = True, 
                                           show_plot= False, save_plot_path=f'outputs/plots/cm/{folder}/{variant}_no_aug')
    best_report_group2 = spatial_evaluate(best_model, model_name = f'{variant}_best model on test group 2', 
                                           loader = test_loader_group2, device=device,
                                           verbose = True, 
                                           plt_confusion_matrix = True, 
                                           show_plot= False, save_plot_path=f'outputs/plots/cm/{folder}/{variant}_no_aug')
    

import matplotlib.pyplot as plt                                                                        
plt.close('all')