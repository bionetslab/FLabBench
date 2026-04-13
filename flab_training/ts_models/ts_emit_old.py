from argparse import Namespace
from .ts_model import TimeSeriesModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CVE(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = int(np.sqrt(args.hid_dim))
        self.W1 = nn.Parameter(torch.empty(1, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, args.hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x):
        # x: bsz, max_len
        x = torch.unsqueeze(x, -1)
        x = torch.matmul(x, self.W1) + self.b1[None, None, :]  # bsz,max_len,int_dim
        x = self.activation(x)
        x = torch.matmul(x, self.W2)  # bsz,max_len,hid_dim
        return x


class FusionAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = args.hid_dim
        self.W = nn.Parameter(torch.empty(args.hid_dim, int_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.u = nn.Parameter(torch.empty(int_dim, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh

    def forward(self, x, mask):
        # x: bsz, max_len, hid_dim
        # mask: bsz, max_len
        att = torch.matmul(x, self.W) + self.b[None, None, :]  # bsz,max_len,int_dim
        att = self.activation(att)
        att = torch.matmul(att, self.u)[:, :, 0]  # bsz,max_len
        att = att + (1 - mask) * torch.finfo(att.dtype).min
        att = torch.softmax(att, dim=-1)  # bsz,max_len
        return att


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.N = args.num_layers
        self.d = args.hid_dim
        self.dff = self.d * 2
        self.attention_dropout = args.attention_dropout
        self.dropout = args.dropout
        self.h = args.num_heads
        self.dk = self.d // self.h
        self.all_head_size = self.dk * self.h

        self.Wq = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wk = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wv = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wo = nn.Parameter(self.init_proj((self.N, self.all_head_size, self.d)), requires_grad=True)
        self.W1 = nn.Parameter(self.init_proj((self.N, self.d, self.dff)), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros((self.N, 1, 1, self.dff)), requires_grad=True)
        self.W2 = nn.Parameter(self.init_proj((self.N, self.dff, self.d)), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros((self.N, 1, 1, self.d)), requires_grad=True)

    def init_proj(self, shape, gain=1):
        x = torch.rand(shape)
        fan_in_out = shape[-1] + shape[-2]
        scale = gain * np.sqrt(6 / fan_in_out)
        x = x * 2 * scale - scale
        return x

    def forward(self, x, mask):
        # x: bsz, max_len, d
        # mask: bsz, max_len
        bsz, max_len, _ = x.size()
        mask = mask[:, :, None] * mask[:, None, :]
        mask = (1 - mask)[:, None, :, :] * torch.finfo(x.dtype).min
        layer_mask = mask
        for i in range(self.N):
            # MHA
            q = torch.einsum('bld,hde->bhle', x, self.Wq[i])
            k = torch.einsum('bld,hde->bhle', x, self.Wk[i])
            v = torch.einsum('bld,hde->bhle', x, self.Wv[i])
            A = torch.einsum('bhle,bhke->bhlk', q, k)
            if self.training:
                dropout_mask = (torch.rand_like(A) < self.attention_dropout
                                ).float() * torch.finfo(x.dtype).min
                layer_mask = mask + dropout_mask
            A = A + layer_mask
            A = torch.softmax(A, dim=-1)
            v = torch.einsum('bhkl,bhle->bkhe', A, v)
            all_head_op = v.reshape((bsz, max_len, -1))
            all_head_op = torch.matmul(all_head_op, self.Wo[i])
            all_head_op = F.dropout(all_head_op, self.dropout, self.training)
            x = (all_head_op + x) / 2
            # FFN
            ffn_op = torch.matmul(x, self.W1[i]) + self.b1[i]
            ffn_op = F.gelu(ffn_op)
            ffn_op = torch.matmul(ffn_op, self.W2[i]) + self.b2[i]
            ffn_op = F.dropout(ffn_op, self.dropout, self.training)
            x = (ffn_op + x) / 2
        return x


class EMIT_TS(TimeSeriesModel):
    """
    EMIT: Event-based Multivariate Irregular Time series model
    
    Key features:
    - Component-level masking (times, values, variables)
    - Dual loss: reconstruction + forecasting
    - Mask tokens for each component
    """
    def __init__(self, args):
        super().__init__(args)
        
        # Embedding layers
        self.cve_time = CVE(args)
        self.cve_value = CVE(args)
        self.variable_emb = nn.Embedding(args.V + 1, args.hid_dim, padding_idx=0)
        
        # Transformer and attention
        self.transformer = Transformer(args)
        self.fusion_att = FusionAtt(args)
        
        # Model parameters
        self.dropout = args.dropout
        self.V = args.V
        self.hid_dim = args.hid_dim
        
        # Mask tokens for component-level masking (EMIT-specific)
        self.time_mask_token = nn.Parameter(torch.randn(1, 1, args.hid_dim), requires_grad=True)
        self.value_mask_token = nn.Parameter(torch.randn(1, 1, args.hid_dim), requires_grad=True)
        self.variable_mask_token = nn.Parameter(torch.randn(1, 1, args.hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.time_mask_token)
        nn.init.xavier_uniform_(self.value_mask_token)
        nn.init.xavier_uniform_(self.variable_mask_token)
        
        # Training mode flags
        self.pretrain = args.train_mode == "pretrain"
        self.finetune = args.train_mode == "finetune"
        
        # Prediction heads
        if self.pretrain or self.finetune:
            self.forecast_head = nn.Linear(self.ts_demo_emb_size, args.V)
        
        if self.finetune:
            self.binary_head = nn.Linear(args.V, 1)
        
        # Error coefficient for masking loss
        self.error_coefficient = getattr(args, 'error_coefficient', 8.0)

    def apply_component_masking(self, time_emb, value_emb, vari_emb, event_mask):
        """
        Apply component-level masking (EMIT-specific)
        Randomly selects one component (time/value/variable) to mask for each event
        
        Args:
            time_emb: (bsz, max_len, hid_dim)
            value_emb: (bsz, max_len, hid_dim)
            vari_emb: (bsz, max_len, hid_dim)
            event_mask: (bsz, max_len) - boolean mask indicating which events to mask
            
        Returns:
            masked embeddings and original embeddings for reconstruction loss
        """
        bsz, max_len, hid_dim = time_emb.size()
        device = time_emb.device
        
        # Randomly choose which component to mask (0: time, 1: value, 2: variable)
        choices = torch.randint(0, 3, (bsz, max_len), device=device)
        
        # Create component-specific masks
        time_choice = (choices == 0).unsqueeze(-1)  # bsz, max_len, 1
        value_choice = (choices == 1).unsqueeze(-1)
        variable_choice = (choices == 2).unsqueeze(-1)
        
        # Expand event_mask for broadcasting
        event_mask_exp = event_mask.unsqueeze(-1)  # bsz, max_len, 1
        
        # Apply masking: replace with mask token where chosen and event_mask is True
        masked_time_emb = torch.where(
            time_choice & event_mask_exp,
            self.time_mask_token.expand(bsz, max_len, hid_dim),
            time_emb
        )
        masked_value_emb = torch.where(
            value_choice & event_mask_exp,
            self.value_mask_token.expand(bsz, max_len, hid_dim),
            value_emb
        )
        masked_vari_emb = torch.where(
            variable_choice & event_mask_exp,
            self.variable_mask_token.expand(bsz, max_len, hid_dim),
            vari_emb
        )
        
        return masked_time_emb, masked_value_emb, masked_vari_emb

    def compute_masking_loss(self, masked_contextual_emb, original_triplet_emb, event_mask):
        """
        Compute reconstruction loss for masked events
        
        Args:
            masked_contextual_emb: (bsz, max_len, hid_dim) - output from transformer
            original_triplet_emb: (bsz, max_len, hid_dim) - original triplet embeddings
            event_mask: (bsz, max_len) - boolean mask
            
        Returns:
            reconstruction loss (scalar)
        """
        # Only compute loss for masked events
        masked_positions = event_mask.unsqueeze(-1)  # bsz, max_len, 1
        
        # Squared error at masked positions
        squared_error = (masked_contextual_emb - original_triplet_emb) ** 2
        masked_error = squared_error * masked_positions
        
        # Average over masked positions
        num_masked = masked_positions.sum() + 1e-8  # Avoid division by zero
        masking_loss = masked_error.sum() / num_masked
        
        return masking_loss

    def forecast_loss(self, ts_emb, forecast_values, forecast_mask):
        """
        Compute forecasting loss
        
        Args:
            ts_emb: (bsz, ts_demo_emb_size) - fused time series + demo embedding
            forecast_values: (bsz, V) - target values
            forecast_mask: (bsz, V) - mask indicating which variables to predict
            
        Returns:
            forecasting loss (scalar)
        """
        pred = self.forecast_head(ts_emb)  # bsz, V
        squared_error = (pred - forecast_values) ** 2
        masked_error = forecast_mask * squared_error
        return masked_error.sum() / (forecast_mask.sum() + 1e-8)

    def forward(self, values, times, varis, obs_mask, demo,
                labels=None, forecast_values=None, forecast_mask=None, event_mask=None):
        """
        Forward pass
        
        Args:
            values: (bsz, max_len) - lab values
            times: (bsz, max_len) - timestamps
            varis: (bsz, max_len) - variable indices
            obs_mask: (bsz, max_len) - padding mask
            demo: (bsz, demo_dim) - demographics
            labels: (bsz,) - binary labels (for finetuning)
            forecast_values: (bsz, V) - forecast targets (for pretraining/finetuning)
            forecast_mask: (bsz, V) - forecast mask (for pretraining/finetuning)
            event_mask: (bsz, max_len) - event mask for component-level masking (for pretraining)
            
        Returns:
            If pretraining: (total_loss, ts_demo_emb)
            Otherwise: (logits, ts_demo_emb)
        """
        bsz, max_len = values.size()
        device = values.device
        
        # Demographics embedding
        demo_emb = self.demo_emb(demo)
        
        # Initial triplet embedding (unmasked)
        time_emb = self.cve_time(times)
        value_emb = self.cve_value(values)
        vari_emb = self.variable_emb(varis)
        
        # Store original for reconstruction loss
        original_triplet_emb = time_emb + value_emb + vari_emb
        
        # Apply component-level masking if in pretraining mode
        if self.pretrain and self.training and event_mask is not None:
            time_emb, value_emb, vari_emb = self.apply_component_masking(
                time_emb, value_emb, vari_emb, event_mask
            )
        
        # Combine masked embeddings
        triplet_emb = time_emb + value_emb + vari_emb
        triplet_emb = F.dropout(triplet_emb, self.dropout, self.training)
        
        # Contextual embedding through transformer
        contextual_emb = self.transformer(triplet_emb, obs_mask)
        
        # Fusion attention
        attention_weights = self.fusion_att(contextual_emb, obs_mask)[:, :, None]
        ts_emb = (contextual_emb * attention_weights).sum(dim=1)
        
        # Concatenate with demographics
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        
        # Compute loss or return predictions
        if self.pretrain:
            # Compute dual loss: masking + forecasting
            masking_loss = self.compute_masking_loss(
                contextual_emb, original_triplet_emb, event_mask
            ) if event_mask is not None else 0.0
            
            forecasting_loss = self.forecast_loss(
                ts_demo_emb, forecast_values, forecast_mask
            )
            
            total_loss = forecasting_loss + self.error_coefficient * masking_loss
            return total_loss, ts_demo_emb
            
        elif self.finetune:
            # Return logits from binary head
            logits = self.binary_head(self.forecast_head(ts_demo_emb))[:, 0]
            return logits, ts_demo_emb
        else:
            # Supervised training without pretraining
            logits = self.binary_head(ts_demo_emb)[:, 0]
            return logits, ts_demo_emb
