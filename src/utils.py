import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from data.splits.sub_splits import TRAIN_SUBS, TEST_SUBS, TEST_SUBS_ALL_STAGES, TEST_SUBS_WAKE_ONLY
import os, json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_split_train_test(test_size, random_seed = 42, train_subs = TRAIN_SUBS, test_subs = TEST_SUBS):
  all_subs = list(train_subs + test_subs)
  train, test = train_test_split(all_subs, test_size = test_size, random_state=random_seed)
  print(f'{len(train)} participants in train/val list; {len(test)} participants in test list.')
  return train, test

def generate_predefined_train_test_loader(fc_dataset, train_val_split, 
                                          sleep_states = {'wake':0,'N1':1,'N2':2,'N3':3}, 
                                          train_val_random_seed = 42, beta = 0.99,
                                          weighted = False, batch_size = 8,
                                          train_subs = TRAIN_SUBS, test_subs = TEST_SUBS):
  g = torch.Generator()
  g.manual_seed(42)

  test_data, train_val_data = [], []
  for i in range(len(fc_dataset.graphs)):
    pt = fc_dataset.graphs[i][0].participant
    if pt in test_subs:
      test_data.append(fc_dataset.graphs[i])
    elif pt in train_subs:
      train_val_data.append(fc_dataset.graphs[i])
  if train_val_split == 0:
    train_data = train_val_data
    val_data = []
  else:
    train_data, val_data = train_test_split(train_val_data, test_size = train_val_split, random_state=train_val_random_seed)

  print(f'{len(train_data)} participants in training set, {len(val_data)} participants in validation set, {len(test_data)} participants in testing set.')
  print('======')

  label_encoder = {v: k for k, v in sleep_states.items()}

  #generate balanced train data list
  weights = []
  data_list = []
  graphs_by_pt_state = {}
  for participant in train_data:
    graphs_by_pt_state[participant[0].participant] = {}
    for graph in participant:
      state = graph.state
      state = label_encoder[state]
      if state not in graphs_by_pt_state[participant[0].participant].keys():
        graphs_by_pt_state[participant[0].participant][state] = []
      weight = (1. - beta)/(1. - beta ** torch.tensor([fc_dataset.graph_class_counts_true[state]]))
      weights.append(weight)
      graphs_by_pt_state[participant[0].participant][state].append(graph)
      data_list.append(graph)

  #generate val data list
  val_list = []
  if not train_val_split == 0:
    for participant in val_data:
      for graph in participant:
        val_list.append(graph)

  #generate test data list
  test_list = []
  for participant in test_data:
    for graph in participant:
      test_list.append(graph)

  #balance train dataset by weighted sampling
  if weighted == True:
    sampled = random.choices(data_list, k=len(data_list), weights = weights)
    train_loader = DataLoader(sampled, batch_size=batch_size, shuffle=True)
    train_loader_deduplicated = DataLoader(data_list, batch_size=batch_size, shuffle=True)
  else:
     train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
     train_loader_deduplicated = train_loader
  test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=True)
  try:
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=True)
  except:
    val_loader = None


  print(f'Total number of unique graphs in training set: {len(data_list)}')
  print(f'Total number of graphs in training set: {len(train_loader)}')
  print(f'Total number of graphs in validation set: {len(val_list)}')
  print(f'Total number of graphs in testing set: {len(test_list)}')

  return train_loader, train_loader_deduplicated, val_loader, test_loader

def generate_test_group_loader(fc_dataset, sleep_states = {'wake':0,'N1':1,'N2':2,'N3':3}, 
                               test_subs_group1 = TEST_SUBS_ALL_STAGES, test_subs_group2 = TEST_SUBS_WAKE_ONLY):
  test_group1, test_group2 = [], []
  for i in range(len(fc_dataset.graphs)):
    pt = fc_dataset.graphs[i][0].participant
    if pt in test_subs_group1:
      test_group1.append(fc_dataset.graphs[i])
    elif pt in test_subs_group2:
      test_group2.append(fc_dataset.graphs[i])


  print(f"{len(test_group1)} participants in testing set group 1, \nthis group contains participants who experienced all 4 brain states.")
  print(f"{len(test_group2)} participants in testing set group 2, \nthis group contains participants who did not fall asleep.")
  print('======')
  
  label_encoder = {v: k for k, v in sleep_states.items()}

  #generate test data list
  test_list1 = []
  for participant in test_group1:
    for graph in participant:
      test_list1.append(graph)
  test_list2 = []
  for participant in test_group2:
    for graph in participant:
      test_list2.append(graph)

  test_loader1 = DataLoader(test_list1, batch_size=256, shuffle=False)
  test_loader2 = DataLoader(test_list2, batch_size=256, shuffle=False)

  print(f'Total number of unique graphs in testing set group 1: {len(test_list1)}')
  print(f'Total number of unique graphs in testing set group 2: {len(test_list2)}')

  return test_loader1, test_loader2

def compute_class_node_feature_property(train_loader, num_nodes = 347, num_features = 20, num_classes = 4):
    # Class-wise node feature augmentation constraints: std, entropy
    class_node_features = defaultdict(lambda: [[] for _ in range(num_nodes)])
    seen = []

    for data in train_loader:
        sample = str(data.participant[0])+'_'+str(int(data.state[0]))+'_'+str(float(data.segment[0]))
        if sample not in seen:
          cls = int(data.state.item())  
          for n in range(num_nodes):
              class_node_features[cls][n].append(data.nodes[n])  # [features]
          seen.append(sample)

    # STD
    class_node_std = {}
    for cls, node_feats in class_node_features.items():
        std_per_node = []
        for feats in node_feats:
            feats_tensor = torch.stack(feats)  # [num_samples, features]
            std = feats_tensor.std(dim=0)
            std_per_node.append(std)
        std_table = torch.stack(std_per_node)  # [nodes, features]
        class_node_std[cls] = std_table # [nodes, features]

    # Entropy
    class_node_entropy = {}
    for cls in range(num_classes):
        entropy_matrix = torch.zeros((num_nodes, num_features))
        for node_idx in range(num_nodes):
            for feat_idx in range(num_features):
                values = class_node_features[cls][node_idx][feat_idx]
                if len(values) > 1:
                    hist = torch.histc(torch.tensor(values).clone().detach(), bins=10)
                    prob = hist / hist.sum()
                    prob = prob[prob > 0]  # avoid log(0)
                    entropy = -(prob * prob.log()).sum().item()
                else:
                    entropy = 0.0  # no variation
                entropy_matrix[node_idx, feat_idx] = entropy
        entropy_matrix = (entropy_matrix - entropy_matrix.min()) / (entropy_matrix.max() - entropy_matrix.min() + 1e-8)
        class_node_entropy[cls] = entropy_matrix

    return class_node_std, class_node_entropy

def augment_node_features(data, class_node_std, class_node_entropy, step=0.1, augment_prob=0.3):
    # Generate random noise under class-wise node feature constraints
    data = data.clone()
    if torch.rand(1).item() < augment_prob:
        cls = int(data.state.item())
        std_table = class_node_std[cls]  # shape: [nodes, features]
        entropy_table = class_node_entropy[cls]  # shape: [nodes, features]
        noise = torch.randn((data.nodes.shape[0], 1)) * step * std_table * entropy_table 
        direction = random.randint(-1,1)
        scale = direction * torch.abs(noise)/(torch.abs(data.nodes) + 1e-6) + 1
        data.nodes = data.nodes * scale
    return data

def save_train_log(logs, variant, root="outputs/logs/train"):
    filename = os.path.join(root, f"{variant}.json")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(logs, f, indent=2)

def save_trained_model(model, variant, variant_name, root = 'models'):
   folder = os.path.join(root, variant)
   os.makedirs(folder, exist_ok=True)
   torch.save(model.state_dict(), os.path.join(folder, f'{variant_name}_weights.pt'))
   torch.save(model, os.path.join(folder, f'{variant_name}.pt'))

def generate_cm(targets, preds, model_name, sleep_states = {'wake':0,'N1':1,'N2':2,'N3':3}, show = False, save_path = None):
    conf_matrix = confusion_matrix(np.array(targets), np.array(preds))

    # Normalize row-wise (by true class counts)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_percentage = np.divide(
        conf_matrix.astype('float'),
        row_sums,
        out=np.zeros_like(conf_matrix, dtype=float),
        where=row_sums != 0
    )

    # Create labels combining count and percentage
    labels = np.array([[f"{percent:.2f}\n({count})" for count, percent in zip(row_counts, row_percents)]
                      for row_counts, row_percents in zip(conf_matrix, conf_matrix_percentage)])

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_percentage, annot=labels, fmt="", cmap="YlGnBu", xticklabels=list(sleep_states.keys()), yticklabels=list(sleep_states.keys()))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")

    if show == True:
       plt.show()

    if save_path:
       os.makedirs(save_path, exist_ok=True)
       plt.savefig(os.path.join(save_path, f"{model_name}_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    
    plt.close()

def generate_data_loaders_from_pt_list(fc_dataset, pt_map, sleep_states = {'wake':0,'N1':1,'N2':2,'N3':3}):

  loaders = {}

  for i in range(len(fc_dataset.graphs)):
    pt = fc_dataset.graphs[i][0].participant
    pt_group = pt_map[pt]
    if pt_group not in loaders.keys():
       loaders[pt_group] = {'pt':[], 'graphs':[], 'loader':None}
    loaders[pt_group]['pt'].append(fc_dataset.graphs[i])

  label_encoder = {v: k for k, v in sleep_states.items()}

  for pt_group in loaders.keys():
    pts = loaders[pt_group]['pt']
    for pt in pts:
      for graph in pt:
        loaders[pt_group]['graphs'].append(graph)
    loaders[pt_group]['loader'] = DataLoader(loaders[pt_group]['graphs'], batch_size=1, shuffle=True)
     
  return loaders
