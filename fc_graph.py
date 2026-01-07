import torch
from torch_geometric.data import Data, DataLoader

def construct_node(fc, roi, exclude_none):
  features = {}
  features['x'] = float(roi.pos[0])
  features['y'] = float(roi.pos[1])
  features['z'] = float(roi.pos[2])
  features['of Network'] = roi.network_id
  for network_id, corr in roi.network_correlations.items():
    network = fc.networks[network_id].network_name
    num_neighbors = roi.num_neighbors_by_network[network_id]
    if num_neighbors != 0:
      corr = float(corr/num_neighbors)
    if exclude_none:
      if network != 'None' and type(network) == str:
        features[network] = corr
    if not exclude_none:
      features[network] = corr
  return features

def construct_vanillaGraphData(fc, save_path, exclude_none = True, save = True): # Without edge filtering, full graph
  data = Data()
  x = []
  node_pos = []
  edge_index = []
  edge_attr = []

  for roi in fc.rois.values():
    if exclude_none:
      if roi.network != 'None' and type(roi.network) == str:
        features = construct_node(fc, roi, exclude_none = exclude_none)
        x.append(torch.tensor(list(features.values())))
        node_pos.append(torch.tensor(list(roi.pos)))
    elif not exclude_none:
      features = construct_node(fc, roi, exclude_none = exclude_none)
      x.append(torch.tensor(list(features.values())))
      node_pos.append(torch.tensor(list(roi.pos)))

  for roi in fc.rois.values():
    if exclude_none:
      if roi.network != 'None' and type(roi.network) == str:
        for neighbor_order in roi.roi_correlations.keys():
            if fc.rois[neighbor_order].network != 'None' and type(fc.rois[neighbor_order].network) == str:
              edge_attr.append([roi.roi_correlations[neighbor_order]])
    elif not exclude_none:
      for neighbor_order in roi.roi_correlations.keys():
        edge_attr.append([roi.roi_correlations[neighbor_order]])

  data.participant = fc.participant
  data.state = fc.state
  data.segment = fc.segment
  data.x = torch.stack(x)
  data.pos = torch.stack(node_pos)
  # FCs are fully connected:
  for i in range(data.x.shape[0]):
    for j in range(data.x.shape[0]):
      edge_index.append([i,j])
  data.edge_index = torch.tensor(edge_index).T
  data.edge_attr = torch.tensor(edge_attr)

  if save:
    torch.save(data, save_path)
  return data