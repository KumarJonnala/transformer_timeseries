# 3.1 TabTransformer model class, modified from Medium

import torch
import torch.nn as nn

class TabTransformer(nn.Module):
    def __init__(self, num_features, num_classes, dim_embedding, num_heads, num_layers, dropout):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, dim_embedding) # project input features -> embedding
        # transformer encoder (batch_first=True so input shape is [batch_size, timesteps, num_features/dim_embedding])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=num_heads,
            dim_feedforward=dim_embedding * 4,
            batch_first=True,
            dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_embedding, num_classes) # simple linear classifier

    def forward(self, x):
        # x: [batch, timesteps, features]
        x = self.embedding(x)            # -> [batch, timesteps, dim_embedding], project input to embedding
        x = self.transformer(x)          # -> [batch, timesteps, dim_embedding], passes through multiple [Attention + FFN + Norm] layers
        x = torch.mean(x, dim=1)         # -> [batch, dim_embedding], global mean pooling over timesteps 
        x = self.classifier(x)           # -> [batch, num_classes], final classification head
        return x

