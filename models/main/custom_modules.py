#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=None, activation_fn=nn.ReLU, 
                 dropout_rate=0.0, use_batch_norm=False, output_activation_fn=None):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        if hidden_dims:
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(h_dim))
                layers.append(activation_fn())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        if output_activation_fn:
            layers.append(output_activation_fn())
            
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        return self.mlp(x)


class GatedModalityFusion(nn.Module):

#     gated fusion for three modalities: Interviewer Question Text (q_text), Participant Answer Text (a_text), Participant Answer Audio (a_audio).

    def __init__(self, input_dim=768, output_dim=768, mlp_hidden_dim=None, mlp_layers=2, activation_fn=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # MLPs for candidate representations (h_m); two-layer MLP with ReLU
        hidden_dims_mlp = [mlp_hidden_dim if mlp_hidden_dim else input_dim] * (mlp_layers - 1) if mlp_layers > 1 else []
        
        self.mlp_q = MLP(input_dim, output_dim, hidden_dims=hidden_dims_mlp, activation_fn=activation_fn)
        self.mlp_t = MLP(input_dim, output_dim, hidden_dims=hidden_dims_mlp, activation_fn=activation_fn)
        self.mlp_a = MLP(input_dim, output_dim, hidden_dims=hidden_dims_mlp, activation_fn=activation_fn)

        self.gate_q = nn.Linear(input_dim, output_dim)
        self.gate_t = nn.Linear(input_dim, output_dim)
        self.gate_a = nn.Linear(input_dim, output_dim)

    def forward(self, q_text_feat, a_text_feat, a_audio_feat):
        
        h_q = self.mlp_q(q_text_feat)
        h_t = self.mlp_t(a_text_feat)
        h_a = self.mlp_a(a_audio_feat)

        g_q = torch.sigmoid(self.gate_q(q_text_feat))
        g_t = torch.sigmoid(self.gate_t(a_text_feat))
        g_a = torch.sigmoid(self.gate_a(a_audio_feat))

        z_j = g_q * h_q + g_t * h_t + g_a * h_a  # Element-wise multiplication (gating) and sum
        return z_j


class DCoPE(nn.Module):

#     Dialogue-based Contextual Positional Encoding (D-CoPE).
#     Combines sinusoidal absolute positional embedding (p_j) with a question context vector (qc_j).
#     Outputs the combined encoding C_j.

    def __init__(self, model_dim=768, qc_mlp_hidden_dim=256, dropout_rate=0.1, max_seq_len=512):
        super().__init__()
        self.model_dim = model_dim
        
        # 768 -> 256 (hidden) -> 768 (output qc_j) with GeLU
        self.mlp_qc = MLP(input_dim=model_dim, 
                          output_dim=model_dim, 
                          hidden_dims=[qc_mlp_hidden_dim], 
                          activation_fn=nn.GELU,
                          dropout_rate=dropout_rate)

        # Linear projection for combining concatenated p_j and qc_j
        self.combination_projection = nn.Linear(2 * model_dim, model_dim)
        
        # For sinusoidal positional encoding p_j
        # Pre-calculate or register buffer if max_seq_len is fixed and known
        pe = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) 

    def get_sinusoidal_pe(self, turn_indices):

    # Generates sinusoidal positional embeddings for given turn indices.

        return self.pe[turn_indices, :]


    def forward(self, q_text_feat, turn_indices):

        # generate Sequential Position Embedding (p_j)

        p_j = self.get_sinusoidal_pe(turn_indices) # (batch_size, seq_len, model_dim)
        
        # Generate Question Context Vector (qc_j)
        
        qc_j = self.mlp_qc(q_text_feat) # (batch_size, seq_len, model_dim)

        # Combine p_j and qc_j
        concatenated_pq = torch.cat((p_j, qc_j), dim=-1) 
        c_j = self.combination_projection(concatenated_pq)
        
        return c_j

class AttentionPooling(nn.Module):

    def __init__(self, input_dim=768, attention_dim=None, dropout_rate=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim if attention_dim is not None else input_dim // 2
        
        self.attention_weights_mlp = MLP(input_dim=self.input_dim,
                                         output_dim=1,
                                         hidden_dims=[self.attention_dim],
                                         activation_fn=nn.Tanh, 
                                         dropout_rate=dropout_rate)

    def forward(self, sequence_output, attention_mask=None):

        attention_scores_e = self.attention_weights_mlp(sequence_output) 
        
        if attention_mask is not None:

            attention_scores_e.masked_fill_(~attention_mask.unsqueeze(-1), -float('inf'))

        attention_weights_alpha = F.softmax(attention_scores_e, dim=1) 
        
        h_global = torch.sum(attention_weights_alpha * sequence_output, dim=1)
        
        return h_global


# Lambda value for GRL 
class GradientReversalLayerFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_val):

        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        
        # Reverse the gradient by multiplying with -lambda_val.    
        # grad_output is the gradient from the subsequent layer (adversarial classifier)
        # We multiply it by -lambda_val before passing it to the preceding layers (main model)
        # The gradient for lambda_val itself is None as it's not a learnable parameter here
        return (grad_output.neg() * ctx.lambda_val), None


if __name__ == '__main__':
    print("--- Testing Custom MLP ---")
    mlp = MLP(input_dim=10, output_dim=2, hidden_dims=[20, 15], activation_fn=nn.Sigmoid, dropout_rate=0.1)
    print(mlp)
    dummy_input_mlp = torch.randn(4, 10)
    output_mlp = mlp(dummy_input_mlp)
    print("MLP Output shape:", output_mlp.shape) 

    print("\n--- Testing GatedModalityFusion ---")
    fusion_module = GatedModalityFusion(input_dim=64, output_dim=64, mlp_hidden_dim=32)
    print(fusion_module)
    dummy_q = torch.randn(4, 10, 64) 
    dummy_a_text = torch.randn(4, 10, 64)
    dummy_a_audio = torch.randn(4, 10, 64)
    output_fusion = fusion_module(dummy_q, dummy_a_text, dummy_a_audio)
    print("Fusion Output shape:", output_fusion.shape)

    print("\n--- Testing DCoPE ---")
    model_dim_test = 64
    dcope_module = DCoPE(model_dim=model_dim_test, qc_mlp_hidden_dim=32, max_seq_len=20)
    print(dcope_module)
    dummy_q_text_feat_dcope = torch.randn(4, 10, model_dim_test)
    dummy_turn_indices = torch.randint(0, 20, (4, 10))
    output_dcope = dcope_module(dummy_q_text_feat_dcope, dummy_turn_indices)
    print("DCoPE Output shape:", output_dcope.shape)

    print("\n--- Testing AttentionPooling ---")
    attn_pooling_module = AttentionPooling(input_dim=model_dim_test, attention_dim=32)
    print(attn_pooling_module)
    dummy_sequence_output = torch.randn(4, 10, model_dim_test) # (batch, seq_len, feature_dim)
    dummy_attention_mask = torch.ones(4, 10, dtype=torch.bool)
    dummy_attention_mask[0, 7:] = False 
    output_attn_pool = attn_pooling_module(dummy_sequence_output, attention_mask=dummy_attention_mask)
    print("AttentionPooling Output shape:", output_attn_pool.shape)

    print("\n--- Testing GradientReversalLayer ---")

    print("GRL is a torch.autograd.Function, tested within a model's backward pass.")

