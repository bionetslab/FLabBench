import torch
import torch.nn as nn
from .ts_model import TimeSeriesModel

class MLP(TimeSeriesModel):

    def __init__(self, args):
        super().__init__(args)
    
        ts_dim = args.T * args.V * len(args.variant)
        input_dim = ts_dim
        
        layers = []
        prev_dim = input_dim
        
        # args.hidden_dims should be a list of hidden layer sizes
        for hidden_dim in args.hid_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(args.dropout))
            prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        

    def forward(self, ts, demo, labels=None):
        batch_size = ts.size(0)
        ts_flat = ts.reshape(batch_size, -1)

        ts_emb = self.net(ts_flat)
        demo_emb = self.demo_emb(demo) 
        
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        
        logits = self.binary_head(ts_demo_emb)[:, 0]
        return logits, ts_demo_emb 

"""
        self.net = nn.Sequential(
            nn.Linear(input_dim, args.hidden_dim_1),
            nn.BatchNorm1d(args.hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim_1, args.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
"""

        

        
