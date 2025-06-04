#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure custom_modules.py is in the same directory or accessible
from custom_modules import MLP, GatedModalityFusion, DCoPE, AttentionPooling, GradientReversalLayerFunction

class MultimodalDepressionNet(nn.Module):
    def __init__(self, config):
        """
        Main model for multimodal depression detection with adversarial debiasing.

        Args:
            config (dict or object): Configuration object/dictionary containing hyperparameters.
                Expected keys:
                - text_embed_dim (int): Dimension of pre-extracted text features (768).
                - audio_raw_embed_dim (int): Dimension of pre-extracted raw audio features (1024).
                - audio_proj_dim (int): Dimension after projecting raw audio features (768).
                - fusion_output_dim (int): Output dimension of the fusion module (768).
                - dcope_qc_mlp_hidden_dim (int): Hidden dim for MLP_CoPE in DCoPE (256).
                - dcope_max_seq_len (int): Max sequence length for pre-calculating D-CoPE sinusoidal PEs.
                - dialogue_transformer_layers (int): Number of layers for Dialogue Transformer (e.g., 2).
                - dialogue_transformer_heads (int): Number of attention heads for Dialogue Transformer (e.g., 8).
                - dialogue_transformer_ffn_dim (int): Inner FFN dimension for Dialogue Transformer (e.g., 2048).
                - dialogue_transformer_dropout (float): Dropout for Dialogue Transformer.
                - attn_pool_intermediate_dim (int): Intermediate dim for Attention Pooling (e.g., 256).
                - pred_head_hidden_dim (int): Hidden dim for the prediction head MLP (e.g., 256).
                - pred_head_dropout (float): Dropout for prediction head.
                - num_depression_classes (int): 1 for binary classification/regression, N for multiclass.
                - prediction_task_type (str): 'binary_classification', 'regression', 'multiclass_classification'.
                - adv_classifier_hidden_dim (int): Hidden dim for adversarial classifier MLP (e.g., 128).
                - num_iqf_classes (int): Number of IQF categories for adversarial task.
                - adv_classifier_dropout (float): Dropout for adversarial classifier.
        """
        super().__init__()
        self.config = config

        # Audio Projection Layer
        self.audio_projection = MLP(
            input_dim=config.audio_raw_embed_dim,
            output_dim=config.audio_proj_dim,
            hidden_dims=[],
            activation_fn=nn.Identity, 
            output_activation_fn=nn.Identity 
        )
        

        # Gated fusion
        self.fusion_module = GatedModalityFusion(
            input_dim=config.text_embed_dim, # all inputs to fusion are same dim
            output_dim=config.fusion_output_dim,
            mlp_hidden_dim=config.fusion_output_dim, 
            mlp_layers=2,
            activation_fn=nn.ReLU
        )

        # Dialogue-based CoPE
        self.dcope_module = DCoPE(
            model_dim=config.fusion_output_dim,
            qc_mlp_hidden_dim=config.dcope_qc_mlp_hidden_dim,
            dropout_rate=config.dialogue_transformer_dropout,
            max_seq_len=config.dcope_max_seq_len
        )

        # Dialogue-Level Transformer Encoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.fusion_output_dim, 
            nhead=config.dialogue_transformer_heads,
            dim_feedforward=config.dialogue_transformer_ffn_dim,
            dropout=config.dialogue_transformer_dropout,
            activation=F.gelu,
            batch_first=True
        )
        self.dialogue_transformer = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=config.dialogue_transformer_layers
        )

        # Attention pooling
        self.attention_pooling = AttentionPooling(
            input_dim=config.fusion_output_dim, # Output dim of transformer
            attention_dim=config.attn_pool_intermediate_dim,
            dropout_rate=config.pred_head_dropout # Reuse dropout
        )

        # Prediction head
        pred_output_activation = None
        if config.prediction_task_type == 'binary_classification':
            pred_output_activation = nn.Sigmoid()
        elif config.prediction_task_type == 'multiclass_classification':
            pred_output_activation = nn.Softmax(dim=-1)

        self.prediction_head = MLP(
            input_dim=config.fusion_output_dim, # H_global dim
            output_dim=config.num_depression_classes,
            hidden_dims=[config.pred_head_hidden_dim],
            activation_fn=nn.ReLU,
            dropout_rate=config.pred_head_dropout,
            output_activation_fn=pred_output_activation
        )

        # Adversarial branch
        self.grl_function = GradientReversalLayerFunction.apply
        self.adversarial_classifier = MLP(
            input_dim=config.fusion_output_dim, 
            output_dim=config.num_iqf_classes,
            hidden_dims=[config.adv_classifier_hidden_dim],
            activation_fn=nn.ReLU,
            dropout_rate=config.adv_classifier_dropout,
            output_activation_fn=nn.Softmax(dim=-1) # IQF prediction
        )

    def forward(self, 
                q_text_features,
                a_text_features,   
                a_audio_raw_features, 
                turn_indices,      
                attention_mask,     
                lambda_grl=0.25      
               ):
        
        batch_size, seq_len, _ = q_text_features.shape


        a_audio_projected = self.audio_projection(a_audio_raw_features) 


        z_j = self.fusion_module(q_text_features, a_text_features, a_audio_projected)


        c_j = self.dcope_module(q_text_features, turn_indices)
        

        z_j_prime = z_j + c_j

  
        src_key_padding_mask = ~attention_mask 
        h_j_sequence = self.dialogue_transformer(z_j_prime, src_key_padding_mask=src_key_padding_mask)

        h_global = self.attention_pooling(h_j_sequence, attention_mask=attention_mask)
        
        # Prediction head
        depression_output = self.prediction_head(h_global)
        if self.config.prediction_task_type == 'binary_classification' or self.config.prediction_task_type == 'regression':
            depression_output = depression_output.squeeze(-1)


        # Apply GRL
        h_j_grl = self.grl_function(h_j_sequence, lambda_grl)
        
        # Adversarial Prediction
        iqf_predictions = self.adversarial_classifier(h_j_grl) # (batch_size, seq_len, num_iqf_classes)
        
        # only compute adversarial loss for non-padded items.

        return depression_output, iqf_predictions


if __name__ == '__main__':
    
    class Config:
        text_embed_dim = 768
        audio_raw_embed_dim = 1024
        audio_proj_dim = 768
        fusion_output_dim = 768
        dcope_qc_mlp_hidden_dim = 256
        dcope_max_seq_len = 100 # Max QA pairs in an interview for PE table
        dialogue_transformer_layers = 2
        dialogue_transformer_heads = 8
        dialogue_transformer_ffn_dim = 2048
        dialogue_transformer_dropout = 0.1
        attn_pool_intermediate_dim = 256
        pred_head_hidden_dim = 256
        pred_head_dropout = 0.3
        # prediction_task_type = 'binary_classification'
        # num_depression_classes = 1 
        prediction_task_type = 'regression' # Example
        num_depression_classes = 1          # For regression, output is 1 continuous value
        # prediction_task_type = 'multiclass_classification'
        # num_depression_classes = 4 # Example for multiclass
        
        adv_classifier_hidden_dim = 128
        num_iqf_classes = 8 # Your 7 IQFs + 'Other'
        adv_classifier_dropout = 0.2

    config = Config()
    model = MultimodalDepressionNet(config)
    print(model)

#     # Dummy Input
#     batch_size = 4
#     seq_len = 50 
    
#     dummy_q_text = torch.randn(batch_size, seq_len, config.text_embed_dim)
#     dummy_a_text = torch.randn(batch_size, seq_len, config.text_embed_dim)
#     dummy_a_audio_raw = torch.randn(batch_size, seq_len, config.audio_raw_embed_dim)
#     dummy_turn_indices = torch.randint(0, config.dcope_max_seq_len, (batch_size, seq_len))
    
#     dummy_attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
#     for i in range(batch_size):
#         actual_len = random.randint(seq_len // 2, seq_len) 
#         dummy_attention_mask[i, :actual_len] = True
#         dummy_turn_indices[i, actual_len:] = 0 

#     print(f"\nInput q_text shape: {dummy_q_text.shape}")
#     print(f"Input attention_mask shape: {dummy_attention_mask.shape} (True for valid, False for pad)")
#     print(f"Example attention_mask[0]: {dummy_attention_mask[0].long().tolist()}")


#     # Test model forward pass
#     lambda_grl_test_value = 0.5
#     try:
#         depression_out, iqf_out = model(
#             dummy_q_text, 
#             dummy_a_text, 
#             dummy_a_audio_raw, 
#             dummy_turn_indices, 
#             dummy_attention_mask,
#             lambda_grl=lambda_grl_test_value
#         )
#         print(f"\nDepression Output Shape: {depression_out.shape}")

        
#         print(f"IQF Prediction Output Shape: {iqf_out.shape}")


#     except Exception as e:
#         print(f"Error during model forward pass test: {e}")
#         import traceback
#         traceback.print_exc()

