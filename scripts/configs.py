import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import utils
from src.dataset import ParticipantFCDataset
from src.model import SleepGNN, CustomGATConv, Swish, Mish
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv, GATv2Conv
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


MODELS = {}

MODELS['baseline'] = {'attention': GATv2Conv, 'activation': nn.GELU()} # attention = Softmax
MODELS['sigmoid+gelu'] = {'attention': partial(CustomGATConv, function=torch.sigmoid), 'activation': nn.GELU()}
MODELS['tanh+gelu'] = {'attention': partial(CustomGATConv, function=torch.tanh), 'activation': nn.GELU()}
MODELS['softmax+tanh'] = {'attention': GATv2Conv, 'activation': nn.Tanh()}
MODELS['sigmoid+tanh'] = {'attention': partial(CustomGATConv, function=torch.sigmoid), 'activation': nn.Tanh()}
MODELS['tanh+tanh'] = {'attention': partial(CustomGATConv, function=torch.tanh), 'activation': nn.Tanh()}
MODELS['softmax+swish'] = {'attention': GATv2Conv, 'activation': Swish()}
MODELS['sigmoid+swish'] = {'attention': partial(CustomGATConv, function=torch.sigmoid), 'activation': Swish()}
MODELS['tanh+swish'] = {'attention': partial(CustomGATConv, function=torch.tanh), 'activation': Swish()}
#MODELS['cos+gelu'] = {'attention': partial(CustomGATConv, function=F.cosine_similarity), 'activation': nn.GELU()}
#MODELS['cos+tanh'] = {'attention': partial(CustomGATConv, function=F.cosine_similarity), 'activation': nn.Tanh()}
#MODELS['cos+swish'] = {'attention': partial(CustomGATConv, function=F.cosine_similarity), 'activation': Swish()}
MODELS['softmax+mish'] = {'attention': partial(CustomGATConv, function=F.cosine_similarity), 'activation': nn.GELU()}
MODELS['softmax+mish'] = {'attention': partial(CustomGATConv, function=F.cosine_similarity), 'activation': nn.Tanh()}
MODELS['softmax+mish'] = {'attention': GATv2Conv, 'activation': Mish()}
MODELS['tanh+mish'] = {'attention': partial(CustomGATConv, function=torch.tanh), 'activation': Mish()}