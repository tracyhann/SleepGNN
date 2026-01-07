import os
import json
import pandas as pd
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from torch_geometric.data import Data
import torch

for filename in os.listdir('data/294_Overlapping'):
    if '.txt' in filename:
        print(filename)
labels = {}
with open ('data/294_Overlapping/Test_Data_List_Overlap_Sliding.txt', 'r') as f:
    for line in f.readlines():
        pairs = line.strip().split('\t')
        labels[pairs[0]] = pairs[1]

with open('labels.json', 'w') as f:
    json.dump(labels, f)

print(len(labels))
print(len(os.listdir('data/294_Overlapping')))

roi_alignment = pd.read_pickle('templates/ROI294_ALIGNMENT (1).pkl')
roi_alignment
print(roi_alignment.columns)
order = roi_alignment['ROI']
order = [o-1 for o in order]

def get_filename(file_path):
  return str(str(file_path).split('/')[-1].replace('.mat', ''))

def get_participant(filename):
  return str(str(filename).split('_')[1]).replace('NonOverlap', '')

def get_segment(filename):
  seg = str(filename).split('_')[-1]
  return int(seg)


def plot_fc(fc, cmap = 'rainbow', title = 'Functional Connectivity Matrix (347x347)', zscore = False):
    np.fill_diagonal(fc, 0)
    if zscore == True:
        z_fc = zscore(fc.flatten())  # shape: (155236,)
        z_fc = z_fc.reshape(int(fc.shape[0])**0.5)       # reshape back to (394, 394)
        plt.figure(figsize=(10, 8))
        plt.imshow(z_fc, cmap=cmap, interpolation='nearest',vmin=np.min(fc), vmax=np.max(fc))
    else:
        plt.figure(figsize=(10, 8))
        plt.imshow(fc, cmap=cmap, interpolation='nearest',vmin=np.min(fc)*0.8, vmax=np.max(fc)*0.8)
    plt.colorbar(label='Connectivity strength')
    plt.title(title)
    plt.xlabel('Region')
    plt.ylabel('Region')
    plt.tight_layout()
    plt.savefig('fc.png')
    plt.close('all')


os.makedirs('graphs', exist_ok=True)
CODES = {'wake':0, 'N1':1, 'N2':2, 'N3':3}
for filepath in tqdm(os.listdir('data/294_Overlapping')):
    if '.mat' in filepath:
        data = loadmat(os.path.join('data/294_Overlapping', filepath))
        filename = get_filename(os.path.join('data/294_Overlapping', filepath))
        pt = get_participant(filename)
        seg = get_segment(filename)
        mapped = pt +'_Segment_'+str(seg)+'_Overlap'
        state = labels[mapped]
        key = list(data.keys())[-1]
        fc = data[key][np.ix_(order, order)].astype(np.float32)
        x = []
        for i in range(fc.shape[0]):
            nfc = fc[i]
            coord = np.array(roi_alignment['(x,y,z)'][i]).astype(np.float32)
            netid = np.array(roi_alignment['Network ID'][i]).astype(np.float32)
            features = np.concatenate(( coord, [netid], nfc), axis=0)
            x.append(features)
        x = np.stack(x, axis=0).astype(np.float32)
        x = torch.tensor(x)

        if pt not in os.listdir('graphs'):
            os.makedirs(f'graphs/{pt}')
        graph = {'x': x, 'pt': str(pt), 'seg': int(seg), 'state': str(state), 'state_label':CODES[str(state)]}
        data = Data(**graph)
        with open(f'graphs/{pt}/{mapped}.pkl', 'wb') as f:
            pickle.dump(graph, f)

    else:
        print(filepath, ' is not a .mat file.')

