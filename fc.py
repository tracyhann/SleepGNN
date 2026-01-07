

class ROI():
  def __init__(self, network_ids, network_id, network, pos, order):
    self.network_id = network_id
    self.network = network
    self.pos = pos
    self.order = order
    self.network_correlations = {}
    self.num_neighbors_by_network = {}
    for net_id in network_ids:
      self.network_correlations[net_id] = 0
      self.num_neighbors_by_network[net_id] = 0
    self.roi_correlations = {}

  def color(self, color):
    self.color = color

  def add_correlation(self, neighbor, correlation):
    self.roi_correlations[neighbor.order] = correlation
    self.network_correlations[neighbor.network_id] += correlation
    self.num_neighbors_by_network[neighbor.network_id] += 1 # FOR AVG

class Network():
  def __init__(self, network_id, network_name):
    self.network_id = network_id
    self.network_name = network_name
    self.rois = []

  def color_code(self, color):
    self.color = color

  def add_roi(self, roi):
    self.rois.append(roi.order)

class FC():
  def __init__(self, participant, state, segment):
    self.participant = participant
    self.state = state
    self.segment = segment
    self.rois = {}
    self.networks = {}

  def add_roi(self, roi):
    self.rois[roi.order] = roi

  def add_network(self, network):
    self.networks[network.network_id] = network