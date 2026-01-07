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
BEST_COMBO = (0.2, 0.001, 0.00001)  #0., 0.001, 0.0001
EPOCHS = 50

folder = '1114_294'

#train_subs, test_subs = utils.random_split_train_test(0.2, random_seed=42)
train_subs, test_subs = TRAIN_SUBS, TEST_SUBS
print(f'{len(train_subs)} participants in training, \n{len(test_subs)} participants in testing.')

fc_dataset = construct_participant_fc_dataset(graphs_folder_path = 'graphs', sleep_states = {'wake':0,'N1':1,'N2':2,'N3':3})
fc_dataset.report_stats() # true dataset stats, inspect true class imbalance and distribution

utils.set_seed(34)

train_loader, train_loader_deduplicated, val_loader, test_loader = utils.generate_predefined_train_test_loader(fc_dataset, 
                                                                train_val_split = 0., beta = 0.999, weighted=False, # 55 subjects for training, no validation
                                                                batch_size=4, train_subs = train_subs, test_subs = test_subs)


test_loader_group1, test_loader_group2 = utils.generate_test_group_loader(fc_dataset)

