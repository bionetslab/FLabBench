import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from flab_training.utils import focal_loss_with_logits


class TimeSeriesModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__() 
        self.args = args

        # set demographics embedding
        self.demo_emb = nn.Sequential(
            nn.Linear(self.args.D, self.args.hid_dim_demo * 2),
            nn.Tanh(),
            nn.Linear(self.args.hid_dim_demo * 2, self.args.hid_dim_demo)
        )

        # set combined embedding size
        if self.args.model_type == 'sand':
            self.ts_demo_emb_size = self.args.hid_dim * self.args.M + self.args.hid_dim_demo
        elif self.args.model_type == 'mlp':
            self.ts_demo_emb_size = self.args.hid_dims[-1] + self.args.hid_dim_demo
        else:
            self.ts_demo_emb_size = self.args.hid_dim + self.args.hid_dim_demo

        # classification head
        if self.args.train_mode != "pretrain":
            self.binary_head = nn.Linear(self.ts_demo_emb_size, 1)
            self.pos_class_weight = torch.tensor(args.pos_class_weight, dtype=torch.float32) if args.pos_class_weight is not None else None


    def compute_loss(self, logits, labels):
        logits = torch.clamp(logits, -20, 20)
        return self.binary_cls_final(logits, labels)

    def predict(self, *args, **kwargs):
        logits, _ = self.forward(*args, **kwargs)
        return torch.sigmoid(logits)

    def binary_cls_final(self, logits, labels):
        # compute loss in training / evaluation mode with ground truth labels
        pos_weight = self.pos_class_weight.to(self.args.device) if self.pos_class_weight is not None else None

        # focal loss
        if self.args.loss_fn == "focal":
            return focal_loss_with_logits(
                logits,
                labels,
                gamma=2.0,
                alpha=None,
                pos_weight=pos_weight
            )

        # BCE loss
        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction='mean'
            )
        
"""
    def load_model_params(self):
        # resolve model config
        if self.model_config is None:
            if hasattr(self.args, "default_params"):
                self.model_config = self.args.default_params
                self.args.logger.write('Default model params loaded.')
            else:
                raise ValueError("No model_config provided and args has no 'default_params'")
        else:
            self.args.logger.write('Custom model params loaded.')

        # set model config params
        for k, v in self.model_config.items():
            setattr(self.args, k, v)
"""