# Load FC matrices into ROI, Network, and FC classes
import numpy as np

class ROI():
  def __init__(self, id, pos,order, network, network_id):
    self.id = id #1-X
    self.pos = pos #ROI position: (x,y,z)
    self.order = order #ROI order in the atlas
    self.network = network #network
    self.network_id = network_id #network_id
    self.corr = {} #with roi v:positive correlation
    self.anti_corr = {} #with roi v:negative correlation (anticorrelation)
    self.no_corr = {} #with roi v:0 correlation
    self.corr_degree = 0
    self.anti_corr_degree = 0
    self.no_corr_degree = 0
    self.total_corr_degree = 0

  def add_correlation(self, roi_neighbor, correlation_value):
    if correlation_value == 0:
      self.no_corr[roi_neighbor] = correlation_value
      self.no_corr_degree += 1
    elif correlation_value > 0:
      self.corr[roi_neighbor] = correlation_value
      self.corr_degree += 1
    elif correlation_value < 0:
      self.anti_corr[roi_neighbor] = correlation_value
      self.anti_corr_degree += 1
    self.total_corr_degree = self.corr_degree + self.anti_corr_degree

  def get_distance_between(roi1, roi2):
    return np.linalg.norm(np.array(roi1.pos) - np.array(roi2.pos))

  def get_total_corr_strength(self):
    return sum(self.corr.values())

  def get_total_anti_corr_strength(self):
    return sum(self.anti_corr.values())

class Network():
  def __init__(self, id, network_name, color):
    self.id = id #1-X
    self.network_name = network_name
    self.roi = [] #list of roi ids
    self.num_roi = 0
    self.color = color

  def add_roi(self, roi):
    self.roi.append(roi.id)
    self.num_roi += 1

  def remove_roi(self, roi):
    self.roi.remove(roi.id)
    self.num_roi -= 1

class FC():
  def __init__(self, roi_network_alignment, network_df, participant, fc_matrix, state, segment):
    self.roi_network_alignment = roi_network_alignment
    self.network_df = network_df
    self.fc_matrix = fc_matrix #ndarray
    self.participant = participant
    self.state = state
    self.segment = segment

    self.networks = {}
    self.num_networks = len(self.network_df)

    self.rois = {}
    self.num_rois = len(self.roi_network_alignment)

    self.no_fc = {}
    self.pos_fc = {}
    self.anti_fc = {}

    self.define_network()
    self.define_roi()
    self.define_fc()

  def define_network(self):
    for i in range(self.num_networks):
      network = Network(id = self.network_df.iloc[i]['Network_ID'],
                        network_name = self.network_df.iloc[i]['Network'],
                        color = self.network_df.iloc[i]['Color'])
      self.networks[network.id] = network

  def define_roi(self):
    for i in range(self.num_rois):
      roi = ROI(id = self.roi_network_alignment.iloc[i]['ROI'],
                pos = self.roi_network_alignment.iloc[i]['(x,y,z)'],
                order = self.roi_network_alignment.iloc[i]['Order'],
                network = self.roi_network_alignment.iloc[i]['Network'],
                network_id = self.roi_network_alignment.iloc[i]['Network ID'])
      self.rois[roi.id] = roi
      self.networks[roi.network_id].add_roi(roi)

  def define_fc(self):
    roi_ids = list(self.rois.keys())

    for id1 in roi_ids:
      for id2 in roi_ids:
        if id1 != id2:

          i = self.rois[id1].order-1
          j = self.rois[id2].order-1

          distance = ROI.get_distance_between(self.rois[id1], self.rois[id2])

          self.rois[id1].add_correlation(roi_neighbor = id2, correlation_value = self.fc_matrix[i][j])

          if self.fc_matrix[i][j] > 0:
            self.pos_fc[(id1,id2)] = {'correlation': self.fc_matrix[i][j], 'distance': distance}
          elif self.fc_matrix[i][j] < 0:
            self.anti_fc[(id1,id2)] = {'correlation': self.fc_matrix[i][j], 'distance': distance}
          elif self.fc_matrix[i][j] == 0:
            self.no_fc[(id1,id2)] = {'correlation': self.fc_matrix[i][j], 'distance': distance}