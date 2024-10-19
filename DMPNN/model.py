import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, aggr




class DMPNN(nn.Module):
    """
    DMPNN model
    Parameters:
    - node_in_feats: int, number of input node features
    - edge_in_feats: int, number of input edge features
    - node_out_feats: int, number of output node features
    - edge_hidden_feats: int, number of hidden edge features
    - num_step_message_passing: int, number of message passing steps
    """

    def __init__(self, node_in_feats, edge_in_feats, node_out_feats, edge_hidden_feats, num_step_message_passing):
        super(DMPNN, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.node_init = nn.Linear(node_in_feats, node_out_feats)
        self.edge_init = nn.Linear(edge_in_feats, edge_hidden_feats)
        edge_network = nn.Sequential(nn.Linear(edge_in_feats, edge_hidden_feats), nn.ReLU(), nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats))
        self.conv = NNConv(node_out_feats, node_out_feats, edge_network, aggr='add')
        self.gru = nn.GRU(node_out_feats, node_out_feats)
   
        nn.init.xavier_normal_(self.node_init.weight)
        nn.init.xavier_normal_(self.edge_init.weight)

        


    def forward(self, data):
        '''
        Forward pass
        '''
        node_attr , edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_attr = F.relu(self.node_init(node_attr)) #(num_nodes, node_out_feats)
        hidden_attr = node_attr.unsqueeze(0)
        for _ in range(self.num_step_message_passing):
            node_attr = F.relu(self.conv(node_attr, edge_index, edge_attr))
            node_attr, hidden_attr = self.gru(node_attr.unsqueeze(0), hidden_attr)
            node_attr = node_attr.squeeze(0)
        return node_attr


class MPNNPredictor(nn.Module):
    def __init__(self, node_in_feats : int , edge_in_feats : int, 
                    node_out_feats:int = 64, edge_hidden_feats:int = 128,
                    num_step_message_passing:int = 6, **kwargs):
        super(MPNNPredictor, self).__init__()
        self.gnn = DMPNN(node_in_feats, edge_in_feats, node_out_feats, edge_hidden_feats, num_step_message_passing)
        self.predictor = nn.Sequential(
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, 1)
        )
        self.pool = aggr.Set2Set(node_out_feats, processing_steps=3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
            x = self.gnn(data)
            x = self.pool(x, data.batch)
            x = self.dropout(x)
            x = self.predictor(x)

            return x
        
if __name__ == "__main__":
    ...