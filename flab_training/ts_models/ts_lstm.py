from argparse import Namespace
from .ts_model import TimeSeriesModel
import torch.nn as nn
import torch

# hid_dim
# num_layers
# dropout

class LSTM_TS(TimeSeriesModel):
    def __init__(self, args):
        super().__init__(args)

        input_dim = args.V * len(args.variant)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=args.hid_dim,
            num_layers=args.num_layers,
            batch_first=True,
            dropout=args.dropout if args.num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, ts, demo, labels=None, return_emb: bool = False):
        _, (h_n, _) = self.lstm(ts)
        ts_emb = h_n[-1]

        ts_emb = self.dropout(ts_emb)

        demo_emb = self.demo_emb(demo)  # shape: (batch, demo_dim)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)

        logits = self.binary_head(ts_demo_emb)[:, 0]
        return logits, ts_demo_emb 