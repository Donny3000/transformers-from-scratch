import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_features: int, num_heads: int = 8):
        super().__init__()

        self.num_features_   = num_features
        self.num_heads_      = num_heads

        # Compute the keys, values and queries for all heads.
        # Combine the key, value, query weights into a single matrix
        heads_feats          = num_heads * num_features
        self.to_keys_        = nn.Linear(in_features=num_features, out_features=heads_feats)
        self.to_values_      = nn.Linear(in_features=num_features, out_features=heads_feats)
        self.to_queries_     = nn.Linear(in_features=num_features, out_features=heads_feats)

        # Define the output linear projection
        self.output_         = nn.Linear(in_features=heads_feats, out_features=num_heads)

        # Scaling factor used to scale the dot products of the query and key
        # so the SoftMax function remains in a region that produces sufficiently
        # large gradients.
        self.scaling_factor_ = torch.sqrt(num_features)
    
    def forward(self, x: torch.Tensor) -> torch.tensor:
        # Input of shape seq. len x batch size x feats.
        t, b, k = x.shape

        batch_heads = b * self.num_heads_
        keys    = self.to_keys_(x).view(   t, batch_heads, k) / (self.num_features_ ** (1/4))
        vals    = self.to_values_(x).view( t, batch_heads, k)
        queries = self.to_queries_(x).view(t, batch_heads, k) / (self.num_features_ ** (1/4))

        # Compute the scaled dot-product
        dp = torch.cat([
            queries[:, i, :].matmul(keys[:, i, :].transpose(0, 1)) for i in range(b)
        ], dim=1)
        scaled_attn = torch.matmul(F.softmax(dp, dim=2), x).view(t, b, self.num_heads_ * k)

        out = self.output_(scaled_attn)

        return out
