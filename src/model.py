# Core PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# PyG
from torch_geometric.nn import (global_mean_pool, global_add_pool, MessagePassing)
from torch_geometric.utils import add_self_loops, subgraph

# Misc
import random 
import numpy as np

class ROIEncoder(torch.nn.Module):
    def __init__(self, roi_conv_type, roi_activation, dropout, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.conv = roi_conv_type
        self.activation = roi_activation
        self.dropout = torch.nn.Dropout(p = dropout)

        for i in range(num_layers):
            if i == 0:
                self.convs.append(self.conv(input_dim, hidden_dim, aggr='mean'))
            elif i == num_layers - 1:
                self.convs.append(self.conv(hidden_dim, input_dim, aggr='mean'))
            else:
                self.convs.append(self.conv(hidden_dim, hidden_dim, aggr='mean'))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        return x
    
class NetworkEncoder(torch.nn.Module):
    def __init__(self, net_conv_type, num_heads, net_activation, dropout, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.conv = net_conv_type
        self.num_heads = num_heads
        self.activation = net_activation
        self.dropout = torch.nn.Dropout(p = dropout)

        for i in range(num_layers):
            if i == 0:
                self.convs.append(self.conv(input_dim, hidden_dim, heads = self.num_heads, concat=False))
            elif i == num_layers - 1:
                self.convs.append(self.conv(hidden_dim * num_heads, input_dim, heads = self.num_heads, concat=False))
            else:
                self.convs.append(self.conv(hidden_dim * num_heads, hidden_dim, heads = self.num_heads, concat=False))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x, (edge_index, alpha) = conv(x, edge_index, return_attention_weights=True)  
            x = self.activation(x)
            x = self.dropout(x)
        return x, (edge_index, alpha)

class MLPClassifier(nn.Module):
    def __init__(self, activation, dropout, input_dim, hidden_dim, num_networks=16, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_networks = num_networks
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.net_feature_convs = nn.Sequential(
                                        nn.Linear(input_dim, num_classes),
                                        #nn.Linear(16, num_classes)
                                    )

        # Combine networks' class-specific contributions
        self.class_combiner = nn.Linear(num_networks, 1)  # operates per class

    def forward(self, x):
        # x shape: [B, N, D] → B=batch, N=num_networks, D=input_dim
        B, N, D = x.shape
        assert N == self.num_networks, f"Expected {self.num_networks} networks, got {N}"

        x = self.net_feature_convs(x)  # [B, N, 4] — each network predicts class logits
        x = x.transpose(1, 2)   # [B, 4, N] — group by class, preserve network contributions
        x = self.class_combiner(x)  # [B, 4, 1]
        x = x.squeeze(-1)      # [B, 4] — final class scores per sample

        return x

class CustomGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, function = torch.tanh, concat=True,
                 dropout=0.0, add_self_loops=True, bias=True):
        super().__init__(node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.function = function
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(heads, 2 * out_channels))  # attention vector per head

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.att_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, return_attention_weights=True):
        if isinstance(x, torch.Tensor):
            x = self.lin(x).view(-1, self.heads, self.out_channels)
        else:
            raise NotImplementedError("Only homogeneous graphs supported.")

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Store edge_index and alpha for later retrieval
        self.edge_index = edge_index  # <-- to return later
        self.alpha = None  # <-- we'll assign this in message()

        out = self.propagate(edge_index, x=x)

        if return_attention_weights:
            return out, (self.edge_index, self.alpha)
        return out

    def message(self, x_i, x_j, index, ptr, size_i):
        try:
            x_cat = torch.cat([x_i, x_j], dim=-1)

            # Compute raw attention score
            alpha = (x_cat * self.att).sum(dim=-1)  
            alpha = self.leaky_relu(alpha)
            alpha = self.function(alpha)    
        except:
            alpha = self.function(x_i, x_j, dim=-1)        
        alpha = self.att_dropout(alpha)
        self.alpha = alpha

        return x_j * alpha.unsqueeze(-1)   

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return super().aggregate(inputs, index, ptr, dim_size)

    def update(self, aggr_out):
        if self.concat:
            out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            out = aggr_out.mean(dim=1)  # average over heads if multi-heads

        if self.bias is not None:
            out += self.bias

        return out
    
    def __deepcopy__(self, memo):
        # fallback to deepcopying state_dict only
        new = type(self)(self.in_channels, self.out_channels, heads = self.heads, function = self.function, concat=True,
                 dropout=self.dropout, add_self_loops = self.add_self_loops)
        new.load_state_dict(copy.deepcopy(self.state_dict()))
        return new

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))  # softplus(x) = log(1 + exp(x))

class ROIToNetworkPool(nn.Module):
    def __init__(self, feature_dim, n_networks, group, use_mlp=False, hidden_dim=32):
        super().__init__()
        self.n_networks = n_networks
        self.use_mlp = use_mlp
        self.group = group
        
        if not use_mlp:
            # Learnable scalar weights per ROI (flat parameter for each ROI)
            self.raw_weights = nn.Parameter(torch.randn(1000))  # max ROI count; we'll slice dynamically
        else:
            # Learnable score per ROI based on features
            self.att_mlp = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x):
        out = []

        for i in range(self.n_networks):
            idx = (self.group == i).nonzero(as_tuple=True)[0]   # ROI indices in group i
            if len(idx) == 0:
                out.append(torch.zeros_like(x[0]))  # blank vector if group is empty
                continue

            x_group = x[idx]   # shape: [n_ROIs_in_group, d]
            
            if self.use_mlp:
                score = self.att_mlp(x_group).squeeze(-1)  # [n_ROIs_in_group]
            else:
                score = self.raw_weights[idx]              # slice from big parameter vector

            a = torch.softmax(score, dim=0)                # [n_ROIs_in_group]
            pooled = (a.unsqueeze(-1) * x_group).sum(dim=0)  # [d]
            out.append(pooled)

        return torch.stack(out, dim=0)  # [n_networks, d]




class WeightedGroupPool(nn.Module):
    """
    Learnable weighted sum from ROIs -> Networks.
    - mask: [N_roi, N_net] binary (1 if ROI i can contribute to net j)
    - roi_emb: [B, N_roi, D]
    Returns:
    - z_net: [B, N_net, D]
    - alpha: [N_roi, N_net] (global weights)
    """
    def __init__(self, num_nodes, num_network, temperature: float = 1.0, add_residual_mean: bool = True):
        super().__init__()
        # One learnable logit per (ROI, Net); masked where invalid
        com = [41, 39, 24, 32, 23, 4, 40, 38, 8, 24, 5, 8, 4, 2, 2]
        self.logits = nn.Parameter(torch.zeros(num_nodes, num_network))
        self.temperature = temperature
        self.add_residual_mean = add_residual_mean

    def forward(self, roi_emb: torch.Tensor, mask: torch.Tensor):
        # roi_emb: [B, N_roi, D]
        B, N, D = roi_emb.shape

        # Masked softmax **across ROIs for each network**
        logits = self.logits / max(1e-6, self.temperature)
        logits = logits.masked_fill(mask == 0, -20.0)  # keep it finite & stable
        # alpha[r, n] sums to 1 over r for each n
        alpha = F.softmax(logits, dim=0).float()                    # [N_roi, N_net]

        # Pool ROIs -> networks
        # z_net[b, n, d] = sum_r alpha[r, n] * roi_emb[b, r, d]
        z_net = torch.einsum('brd,rn->bnd', roi_emb, alpha).float()

        return z_net, alpha
    


class SleepGNN(torch.nn.Module):
    def __init__(self, roi_conv_type, net_conv_type, 
                 num_roi_layers, num_network_layers, 
                 input_dim, hidden_dim, 
                 num_nodes = 347, num_network = 16, output_num_classes = 4, 
                 num_heads = 1, edge_retaintion = [0.3,0.6], same_network_only = True,
                 edge_filter = None, activation = nn.GELU(), dropout = 0.5, device = 'cpu'):
        super(SleepGNN, self).__init__()

        self.num_roi_layers = num_roi_layers
        self.num_network_layers = num_network_layers
        self.output_num_classes = output_num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_network = num_network
        self.activation = activation
        self.dropout = torch.nn.Dropout(p = dropout)
        self.edge_retaintion = edge_retaintion
        self.same_network_only = same_network_only

        self.roi_conv = roi_conv_type
        self.net_conv = net_conv_type
        self.edge_filter = edge_filter

        self.edge_mask = torch.nn.Linear(num_nodes, num_nodes)

        self.roi_convs = ROIEncoder(roi_conv_type, activation, dropout, 
                                    input_dim, hidden_dim, num_network, num_roi_layers)
        self.network_convs = NetworkEncoder(net_conv_type, num_heads, activation, dropout, 
                                            input_dim, hidden_dim, num_network, num_network_layers)

        #self.pool = global_add_pool # (16,20)
        #self.pool = global_mean_pool
        self.pool = WeightedGroupPool(num_nodes, num_network)

        self.classifier = MLPClassifier(activation = nn.GELU(), dropout = dropout, input_dim=input_dim, hidden_dim=hidden_dim,
                                        num_networks=self.num_network, num_classes=self.output_num_classes)
        
        '''# Class-specific refinement heads (each sees shared + class probs)
        self.refiners = nn.ModuleList([
            nn.Linear(input_dim + output_num_classes, 1) for _ in range(output_num_classes)
        ])'''

        self.device = device

    '''
    def generate_ROIGraph_edge_index(self, data):
        all_edge_index = data.edge_index
        connectivity = data.edge_connectivity.float()
        positive_mask = (connectivity > 0).view(-1)
        positive_edge_index = all_edge_index[:, positive_mask]
        negative_mask = (connectivity < 0).view(-1)
        negative_edge_index = all_edge_index[:, negative_mask]
        return all_edge_index, positive_edge_index, negative_edge_index
    
    '''
    def generate_ROIGraph_edge_index(self, data, batch_size, same_network_only: bool = True):
        """
        Build per-graph full ROI edge_index and masks, then concat across the batch.

        Assumptions:
        - Each graph has exactly self.num_nodes nodes (e.g., 450).
        - data.fc is either:
            • a list/tuple of length B with per-graph (N,N) arrays/tensors, OR
            • a tensor of shape [B, N, N].
        - 'com' is fixed (same per graph) to define network labels per node.

        Returns:
        roi_edge_index_all : LongTensor [2, B*N*N]          (all full edges, batched)
        pos_edge_index_all : LongTensor [2, total_pos]      (sign>0, optionally same-net)
        neg_edge_index_all : LongTensor [2, total_neg]      (sign<0, optionally same-net)
        same_mask_all      : BoolTensor [B*N*N]             (same-network mask per edge)
        """
        device = self.device
        N = self.num_nodes
        B = data.batch_size

        # --- labels per node (one graph) from com ---
        com = [41, 39, 24, 32, 23, 4, 40, 38, 8, 24, 5, 8, 4, 2, 2]
        labels_one = torch.empty(N, dtype=torch.long, device=device)
        idx = 0
        for net_id, size in enumerate(com):
            labels_one[idx:idx+size] = net_id
            idx += size
        # same-network mask for one graph on the full grid (N x N)
        same_one = (labels_one[:, None] == labels_one[None, :]).reshape(-1)  # [N*N], bool

        # --- base full roi edge grid for one graph ---
        nodes = torch.arange(N, device=device, dtype=torch.long)
        base_edges = torch.cartesian_prod(nodes, nodes).T  # [2, N*N]

        # storage
        edge_blocks = []
        pos_blocks  = []
        neg_blocks  = []
        same_blocks = []

        # helper to get per-graph fc matrix as tensor [N,N] on device
        def get_fc(b):
            D = data.x.shape[-1]
            x = data.x.reshape(B, N, D).to(self.device)
            fc = x[b][:, 4:]
            fc_t = torch.as_tensor(fc, device=device)
            if fc_t.numel() != N*N:
                raise ValueError(f"fc[{b}] has shape {tuple(fc_t.shape)}, expected ({N},{N})")
            return fc_t

        # build per-graph and concat
        for b in range(B):
            offset = b * N
            ei_b = base_edges + offset  # [2, N*N]
            edge_blocks.append(ei_b)

            # connectivity flattened to align with base_edges order
            fc_b = get_fc(b).reshape(-1)  # [N*N]

            if same_network_only:
                same_b = same_one  # reuse
            else:
                same_b = torch.ones_like(same_one, dtype=torch.bool, device=device)

            same_blocks.append(same_b)

            # masks & indices
            pos_mask = (fc_b > 0) & same_b
            neg_mask = (fc_b < 0) & same_b

            if pos_mask.any():
                pos_blocks.append(ei_b[:, pos_mask])
            if neg_mask.any():
                neg_blocks.append(ei_b[:, neg_mask])

        # concat across batch
        roi_edge_index_all = torch.cat(edge_blocks, dim=1) if edge_blocks else torch.empty(2, 0, dtype=torch.long, device=device)
        same_mask_all = torch.cat(same_blocks, dim=0) if same_blocks else torch.empty(0, dtype=torch.bool, device=device)

        pos_edge_index_all = torch.cat(pos_blocks, dim=1) if len(pos_blocks) > 0 else torch.empty(2, 0, dtype=torch.long, device=device)
        neg_edge_index_all = torch.cat(neg_blocks, dim=1) if len(neg_blocks) > 0 else torch.empty(2, 0, dtype=torch.long, device=device)

        return roi_edge_index_all, pos_edge_index_all, neg_edge_index_all, same_mask_all


    
    def generate_NetGraph_edge_index(self):
        retain_low_bound = self.edge_retaintion[0]
        retain_upper_bound = self.edge_retaintion[1]
        netGraph_edge_index = []
        for i in range(self.num_network):
          for j in range(self.num_network):
            netGraph_edge_index.append([i,j])
        netGraph_edge_index = torch.tensor(netGraph_edge_index).T.contiguous()

        '''# Choose a random retention rate between 20% and 60%
        retain_ratio = random.uniform(retain_low_bound, retain_upper_bound)
        num_edges = netGraph_edge_index.size(1)

        # Random permutation of edge indices
        perm = torch.randperm(num_edges)

        # Keep only the top k randomly selected edges
        k = int(retain_ratio * num_edges)
        keep_indices = perm[:k]

        # Apply to edge index and weights
        netGraph_edge_index = netGraph_edge_index[:, keep_indices]'''

        return netGraph_edge_index


    def batchify_edge_index(self, edge_index_single, batch_size, num_nodes_per_graph):
        edge_indices = []
        for i in range(batch_size):
            offset = i * num_nodes_per_graph
            edge_i = edge_index_single + offset
            edge_indices.append(edge_i)
        return torch.cat(edge_indices, dim=1)
    
    def mask_from_labels(self, batch_size):
        """
        group: 1D LongTensor/list, length n_roi, values in [0, n_net-1] or -1 for 'exclude'
        """
        # Community sizes per network
        com = [41, 39, 24, 32, 23, 4, 40, 38, 8, 24, 5, 8, 4, 2, 2]
        labels_per_node = []
        for net_id, size in enumerate(com):
            label = np.zeros(self.num_network)
            label[net_id] = 1
            labels_per_node.extend([label]*size)
        labels_per_node = torch.tensor(labels_per_node, device=self.device)  
        return labels_per_node
    


    def forward(self, data):
        x = data.x.float().to(self.device)
        #edge_connectivity =  data.edge_connectivity.reshape(self.num_nodes, self.num_nodes).float().to(self.device)
        batch = data.batch.to(self.device)
        batch_size = int(data.batch.max().item()) + 1
        mask = self.mask_from_labels(data.batch_size)
        x = x[:, 3:] # exclude ROI position coordinates
        network_id = x[:, 0] # network membership of ROIs
        x = x[:, 1:] # ROI features
        _, group = network_id.unique(return_inverse=True) 
        group = group + batch * self.num_network  # Make group unique per graph
        B, N, D = batch_size, self.num_nodes, x.shape[-1]


        edge_index_group = self.generate_ROIGraph_edge_index(data, self.same_network_only)
        if self.edge_filter == 'pos':
            edge_index = edge_index_group[1].to(self.device)
        elif self.edge_filter == 'neg':
            edge_index = edge_index_group[2].to(self.device)
        else:
            edge_index = edge_index_group[0].to(self.device)
        '''
        #edge_index_mask = (torch.sigmoid(self.edge_mask(edge_connectivity)) > 0.5).view(-1)
        edge_index_mask = (torch.sigmoid(self.edge_mask(torch.ones(self.num_nodes, self.num_nodes))) > 0.5).view(-1)
        edge_index = edge_index[:, edge_index_mask]

        '''        
        x = F.normalize(x, p=2, dim=1)  # row-wise L2 normalization

        # if within network roi communication

        '''# This will hold the updated node embeddings
        x_updated = torch.zeros_like(x).to(self.device)

        unique_group_ids = group.unique()
        for gid in unique_group_ids:
            mask = group == gid
            node_indices = mask.nonzero(as_tuple=False).view(-1).to(self.device)

            # Extract subgraph: only nodes and edges within this network, within network communication
            edge_sub, _ = subgraph(node_indices, edge_index, relabel_nodes=True)
            x_sub = x[node_indices]  # node features
            x_sub_out = self.roi_convs(x_sub, edge_sub) # [N, 128]
            x_updated[node_indices] = x_sub_out

        roi_graph_emb = x_updated'''

        roi_graph_emb = self.roi_convs(x, edge_index)
        roi_graph_emb = roi_graph_emb.reshape(batch_size, self.num_nodes, -1).to(self.device)  # [B, N, D]

        # input_net_graph_emb = self.pool(roi_graph_emb, group).float()
        input_net_graph_emb, alpha = self.pool(roi_graph_emb, mask)
        input_net_graph_emb = input_net_graph_emb.reshape(B*self.num_network, self.input_dim).to(self.device)


        net_graph_edge_index = self.generate_NetGraph_edge_index()
        
        single_net_graph_edge_index = net_graph_edge_index.to(input_net_graph_emb.device)
        batch_net_graph_edge_index = self.batchify_edge_index(single_net_graph_edge_index, batch_size, self.num_network)
        batch_net_graph_edge_index = batch_net_graph_edge_index.to(self.device)
        output_net_graph_emb, _ = self.network_convs(input_net_graph_emb, batch_net_graph_edge_index)

        output_net_graph_emb = output_net_graph_emb.view(batch_size, self.num_network, -1)
        out = self.classifier(output_net_graph_emb)

        return out, (input_net_graph_emb, output_net_graph_emb), net_graph_edge_index