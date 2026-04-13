from argparse import Namespace
from .ts_model import TimeSeriesModel
import torch.nn as nn
import torch

# MODEL PARAMS:
# hid_dim
# num_layers
# dropout


class GRU_TS(TimeSeriesModel):
    def __init__(self, args):
        super().__init__(args)

        # number of features depend on variant V, VM, VMD
        input_dim = args.V * len(args.variant) 

        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(args.hid_dim),   
            num_layers=int(args.num_layers),
            batch_first=True,
            dropout=args.dropout if args.num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, ts, demo, labels=None): #return_emb: bool=False
        # ts: (batch, seq_len, input_dim)
        _, h_n = self.gru(ts)  # h_n: (num_layers, batch, hidden_size)

        ts_emb = h_n[-1]  # shape: (batch, hidden_size)

        # option 2: Concatenate all hidden states across layers (optional)
        # ts_emb = h_n.transpose(0, 1).reshape(ts.size(0), -1)

        ts_emb = self.dropout(ts_emb)

        demo_emb = self.demo_emb(demo)  # shape: (batch, demo_dim)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)

        logits = self.binary_head(ts_demo_emb)[:, 0]
        return logits, ts_demo_emb 
        
        #self.binary_cls_final(logits, labels)
