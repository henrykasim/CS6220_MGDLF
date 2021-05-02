'''Implementation of MGGCN'''

import argparse
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F


# build a Three-layer GCN with ReLU as the activation in between
class GCN(nn.Module):
    def __init__(self, graph, in_feats, h1_feats, h2_feats, h3_feats):
        super(GCN, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h1_feats)
        self.gcn_layer2 = GraphConv(h1_feats, h2_feats)
        self.gcn_layer3 = GraphConv(h1_feats, h2_feats)
        self.graph = graph
    
    def forward(self, inputs):
        #print(self.graph.num_nodes,inputs.shape)
        h = self.gcn_layer1(self.graph, inputs)
        h = F.relu(h)
        h = self.gcn_layer2(self.graph, h)
        h = F.relu(h)
        h = self.gcn_layer3(self.graph, h)
        return h

# GCN+gru for single view
class MGGCN_Cell(nn.Module):
    def __init__(self, graph, in_feats, h1_feats, h2_feats, h3_feats, hidden_size):
        super(MGGCN_Cell, self).__init__()
        self.GCN = GCN(graph, in_feats, h1_feats, h2_feats, h3_feats)
        self.num_nodes = graph.number_of_nodes()
        self.graph = graph
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size = self.num_nodes*h3_feats, hidden_size = self.num_nodes*self.hidden_size)
        self.h3_feats = h3_feats

    
    def forward(self, inputs):
        ##input size (batch,number of nodes,N=2,time interval)
        batch_size = inputs.shape[1]
        time_interval = inputs.shape[3]
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        for i in range(time_interval):
            if i == 0:
                hidden_GCN = self.GCN(inputs[:,:,:,i]).unsqueeze(-1)
            else:
                hidden_GCN = torch.cat((hidden_GCN, self.GCN(inputs[:,:,:,i]).unsqueeze(-1)), 3)
        hidden_GCN = hidden_GCN.transpose(0,1)
        hidden_GCN = hidden_GCN.reshape(batch_size,self.num_nodes*self.h3_feats,time_interval)
        #hidde_GCN = hidden_GCN.permute(2,0,1)
        #print(hidden_GCN.permute(2,0,1).is_cuda)
        _,out = self.gru(hidden_GCN.permute(2,0,1))
        #print('gru output:',torch.max(out).item())
        #print(out.size())
        out = out.squeeze().reshape(batch_size,self.num_nodes,self.hidden_size)
        return out

#fusion of multi-view gcn cell
class MGGCN(nn.Module):
    def __init__(self, graph, in_feats, h1_feats, h2_feats, h3_feats, interval, hidden_size):
        super(MGGCN, self).__init__()
        #recent hourly branch
        self.GCN_r = MGGCN_Cell(graph, in_feats, h1_feats, h2_feats, h3_feats, hidden_size)
        #daily branch
        self.GCN_d = MGGCN_Cell(graph, in_feats, h1_feats, h2_feats, h3_feats, hidden_size)
        #weekly branch
        self.GCN_w = MGGCN_Cell(graph, in_feats, h1_feats, h2_feats, h3_feats, hidden_size)
        #monthly branch
        self.GCN_m = MGGCN_Cell(graph, in_feats, h1_feats, h2_feats, h3_feats, hidden_size)
        self.num_nodes = graph.number_of_nodes()
        #parametric matrix for multi-view fusion
        self.W_r = nn.Parameter(torch.FloatTensor(self.num_nodes,hidden_size))
        self.W_d = nn.Parameter(torch.FloatTensor(self.num_nodes,hidden_size))
        self.W_w = nn.Parameter(torch.FloatTensor(self.num_nodes,hidden_size))
        self.W_m = nn.Parameter(torch.FloatTensor(self.num_nodes,hidden_size))
        self.linear = nn.Linear(hidden_size,2)
        self.graph = graph
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.W_d)
        nn.init.xavier_uniform_(self.W_w)
        nn.init.xavier_uniform_(self.W_m)
        
    def forward(self, inputs):
        O_r = self.GCN_r(inputs[:,:,:,:,0])
        O_d = self.GCN_d(inputs[:,:,:,:,1])
        O_w = self.GCN_w(inputs[:,:,:,:,2])
        O_m = self.GCN_m(inputs[:,:,:,:,3])
        O =  O_r * self.W_r.expand_as(O_r) + O_d * self.W_d.expand_as(O_d) + O_w * self.W_w.expand_as(O_w) + O_m * self.W_m.expand_as(O_m)
        #print('mggcn output before tanh:',torch.max(O).item())
        O = torch.tanh(self.linear(O))
        return O
