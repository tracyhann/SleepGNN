from torch_geometric.data import Data
from torch.utils.data import Dataset
import pickle, os
from tqdm import tqdm
from data.splits.sub_splits import TRAIN_SUBS, TEST_SUBS, TEST_SUBS_ALL_STAGES, TEST_SUBS_WAKE_ONLY
import torch

class ParticipantFCDataset(Dataset):
  def __init__(self, graphs_folder_path = 'graphs', sleep_states = {'wake':0,'N1':1,'N2':2,'N3':3}):
    self.graphs_folder_path = graphs_folder_path
    self.label_encoder = sleep_states
    self.participants = []
    self.states = []
    self.graphs = []
    self.transform = None
    self.graph_class_counts_true = {}

    for state in self.label_encoder.keys():
      self.graph_class_counts_true[state] = 0

    #{'x': fc, 'pt': str(pt), 'seg': int(seg), 'state': str(state), 'state_label':CODES[str(state)]}

    for pt_folder in tqdm(os.listdir(self.graphs_folder_path)[:]):
      pt_folder_path =  os.path.join(self.graphs_folder_path, pt_folder)
      series = []
      for pkl_file in os.listdir(pt_folder_path):
        if pkl_file.endswith('.pkl'):
          with open(os.path.join(pt_folder_path, pkl_file), 'rb') as f:
            data = pickle.load(f)
            participant = data['pt']
            if participant in TRAIN_SUBS or TEST_SUBS:
              self.participants.append(participant)
              data['participant'] = participant
              graph = Data(**data)
              self.graph_class_counts_true[graph.state] += 1
              graph.state = self.label_encoder[graph.state]
              series.append(graph)
            else:
               print(f'Skipping participant {participant}, not in train or test subs.')
      self.graphs.append(series)

    self.stats()

  def stats(self):
    self.num_participants = len(set(self.participants))
    self.num_states = len(set(list(self.graph_class_counts_true.keys())))
    self.num_data = sum(self.graph_class_counts_true.values())

  def __len__(self):
        return len(self.graphs)

  def __getitem__(self, idx):
      sample = self.graphs[idx]
      if self.transform:
          sample = self.transform(sample)
      return sample

  def report_stats(self):
    print(f'Number of participants: {self.num_participants}')
    print(f'Number of states: {self.num_states}')
    print(f'Number of graphs: {self.num_data}')
    print(f'Number of graphs per state (true distribution): \n {self.graph_class_counts_true}')


def construct_participant_fc_dataset(graphs_folder_path = 'graphs', sleep_states = {'wake':0,'N1':1,'N2':2,'N3':3}):
  pt_fc_dataset = ParticipantFCDataset(graphs_folder_path = 'graphs', sleep_states = {'wake':0,'N1':1,'N2':2,'N3':3})
  pt_fc_dataset.report_stats()
  return pt_fc_dataset

'''
with open("graphs/pt_fc_graph_dataset.pkl", "wb") as f:
    pickle.dump(pt_fc_dataset, f)
'''