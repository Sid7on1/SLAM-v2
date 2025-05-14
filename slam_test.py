# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
import math
import warnings
from typing import Optional, Tuple, List, Union, Dict

def generate_synthetic_data(num_samples, seq_len, d_model, num_classes):
    """Generate synthetic data for training and evaluation.
    
    Args:
        num_samples: Number of samples to generate
        seq_len: Sequence length for each sample
        d_model: Dimension of the model
        num_classes: Number of classes for classification
        
    Returns:
        X: Input data of shape [num_samples, seq_len, d_model]
        y: Labels of shape [num_samples]
    """
    # Generate random input sequences
    X = torch.randn(num_samples, seq_len, d_model)
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y

# Model wrapper for classification tasks
class TransformerClassifier(nn.Module):
    """Transformer-based classifier that uses an encoder and a classification head."""
    
    def __init__(self, encoder, d_model, num_classes, seq_len):
        super().__init__()
        self.encoder = encoder
        
        # Classification head - for most models we use CLS token (index 0)
        self.classifier = nn.Linear(d_model, num_classes)
        self.seq_len = seq_len
        
        # Positional encoding for input (needed for models without built-in position encoding)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Pass through encoder with mask
        encoded = self.encoder(x, mask)
        
        # Global average pooling over sequence dimension or use the CLS token
        # Here we use the first token (CLS token approach)
        cls_representation = encoded[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_representation)
        
        return logits

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # Handle mask without reshaping it to avoid dimension mismatches
        if mask is not None:
            if mask.dim() == 2:  # [batch_size, seq_len]
                # Create a simple broadcasted mask for each token
                attn_scores.masked_fill_(
                    mask.unsqueeze(1).unsqueeze(2).eq(0), 
                    -1e9
                )
            elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                # Expand the mask to fit the head dimension automatically
                attn_scores.masked_fill_(
                    mask.unsqueeze(1).eq(0),
                    -1e9
                )
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class StandardTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class SLAMv1(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, ef_cycles=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.ef_cycles = ef_cycles
        
        # Initial full MHSA layer
        self.initial_block = EncoderBlock(d_model, num_heads, d_ff, dropout)
        
        # EF Cycle blocks
        self.ef_blocks_A = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(ef_cycles)
        ])
        
        self.ef_blocks_B = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(ef_cycles)
        ])
        
        # Final refinement block
        self.final_block = EncoderBlock(d_model, num_heads, d_ff, dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Handle mask properly for subsequences
        # Note: We don't reshape the mask, we keep it in its original format
        
        # Initial full MHSA layer
        x = self.initial_block(x, mask)
        
        # EF Cycles
        for i in range(self.ef_cycles):
            # Calculate segment indices - ensure these are integers
            first_60_percent = int(seq_len * 0.6)
            last_60_percent = int(seq_len * 0.6)
            start_40_percent = int(seq_len * 0.4)  # Use explicit int conversion instead of seq_len - last_60_percent
            
            # Create masked versions of the original mask for the segments
            if mask is not None:
                if mask.dim() == 3:  # Attention mask [batch, seq, seq]
                    # Extract submasks for each segment
                    mask_A1 = mask[:, :first_60_percent, :first_60_percent]
                    mask_A2 = mask[:, start_40_percent:, start_40_percent:]
                    mask_B1 = mask[:, start_40_percent:, start_40_percent:]
                    mask_B2 = mask[:, :first_60_percent, :first_60_percent]
                elif mask.dim() == 2:  # Token mask [batch, seq]
                    # Create simple token masks for segments
                    mask_A1 = mask[:, :first_60_percent]
                    mask_A2 = mask[:, start_40_percent:]
                    mask_B1 = mask[:, start_40_percent:]
                    mask_B2 = mask[:, :first_60_percent]
            else:
                mask_A1 = mask_A2 = mask_B1 = mask_B2 = None
            
            # Block A processes first 60% and last 60% segments
            segment_A1 = x[:, :first_60_percent, :]
            segment_A2 = x[:, start_40_percent:, :]
            segment_A = torch.cat([segment_A1, segment_A2], dim=1)
            
            # Block B processes last 60% and first 60% segments (order reversed from A)
            segment_B1 = x[:, start_40_percent:, :]
            segment_B2 = x[:, :first_60_percent, :]
            segment_B = torch.cat([segment_B1, segment_B2], dim=1)
            
            # Create masks for the concatenated segments
            if mask is not None and mask.dim() == 3:
                # For attention masks, we need to handle the cross-attention between segments
                segments_len_A = segment_A.size(1)
                segments_len_B = segment_B.size(1)
                
                mask_A = torch.zeros(batch_size, segments_len_A, segments_len_A, device=x.device)
                mask_A[:, :first_60_percent, :first_60_percent] = mask_A1
                mask_A[:, first_60_percent:, first_60_percent:] = mask_A2
                
                mask_B = torch.zeros(batch_size, segments_len_B, segments_len_B, device=x.device)
                mask_B[:, :last_60_percent, :last_60_percent] = mask_B1
                mask_B[:, last_60_percent:, last_60_percent:] = mask_B2
            elif mask is not None and mask.dim() == 2:
                # For token masks, we can simply concatenate
                mask_A = torch.cat([mask_A1, mask_A2], dim=1)
                mask_B = torch.cat([mask_B1, mask_B2], dim=1)
            else:
                mask_A = mask_B = None
            
            # Process segments through respective blocks
            processed_A = self.ef_blocks_A[i](segment_A, mask_A)
            processed_B = self.ef_blocks_B[i](segment_B, mask_B)
            
            # Split processed segments back to original positions
            processed_A1 = processed_A[:, :first_60_percent, :]
            processed_A2 = processed_A[:, first_60_percent:, :]
            
            processed_B1 = processed_B[:, :last_60_percent, :]
            processed_B2 = processed_B[:, last_60_percent:, :]
            
            # Create new tensor to store the fusion results
            x_new = torch.zeros_like(x)
            counts = torch.zeros((batch_size, seq_len, 1), device=x.device)
            
            # Non-overlapping region from A1: [0, start_40_percent)
            x_new[:, :start_40_percent, :] = processed_A1[:, :start_40_percent, :]
            counts[:, :start_40_percent, :] += 1
            
            # Overlapping region: [start_40_percent, first_60_percent)
            # Average the two outputs in the overlapping region
            overlap_start = start_40_percent
            overlap_end = first_60_percent
            
            x_new[:, overlap_start:overlap_end, :] += processed_A1[:, overlap_start:overlap_end, :]
            counts[:, overlap_start:overlap_end, :] += 1
            
            x_new[:, overlap_start:overlap_end, :] += processed_B2[:, overlap_start:overlap_end, :]
            counts[:, overlap_start:overlap_end, :] += 1
            
            # Non-overlapping region from B1: [first_60_percent, seq_len)
            x_new[:, first_60_percent:, :] = processed_B1[:, first_60_percent-start_40_percent:, :]
            counts[:, first_60_percent:, :] += 1
            
            # Compute the average in the overlapping regions
            # Ensure no division by zero
            counts = counts.clamp(min=1.0)
            x = x_new / counts
        
        # Final refinement block
        x = self.final_block(x, mask)
        
        return x

# DeepSeek Transformer implementation
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Ensure dim is even
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")
        
        # Create the frequency base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Create position indices
        self.register_buffer(
            "pos_ids", torch.arange(max_seq_len, dtype=torch.float)
        )
        
        # Initialize cached cos and sin tables
        self._update_cos_sin_tables()
    
    def _update_cos_sin_tables(self):
        # Calculate cos and sin values for the positions
        freqs = torch.outer(self.pos_ids, self.inv_freq)
        # Two different frequency sets - alternative arrangement
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=torch.float))
        self.register_buffer("sin_cached", emb.sin().to(dtype=torch.float))
    
    def forward(self, x, seq_len=None):
        # Return the cached values for the positions we need
        if seq_len is None:
            seq_len = min(x.shape[1], self.max_seq_len)
        else:
            seq_len = min(seq_len, self.max_seq_len)
            
        # Return appropriately sized cos and sin embeddings
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embeddings to q and k tensors.
    
    Args:
        q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim] 
           or [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor of same shape as q
        cos: Cosine part of rotary embeddings
        sin: Sine part of rotary embeddings
        position_ids: Optional custom position ids
        
    Returns:
        q_embed: Query with rotary embeddings applied
        k_embed: Key with rotary embeddings applied
    """
    # Check if tensors are in [batch, seq_len, heads, dim] format and transpose if needed
    q_orig_shape = q.shape
    k_orig_shape = k.shape
    
    # Ensure q and k are in [batch, seq, head, dim] format for applying rotary embeddings
    if len(q_orig_shape) == 4 and q_orig_shape[1] == q_orig_shape[2] and q_orig_shape[2] == q_orig_shape[1]:
        # If q is in [batch, head, seq, dim] format, transpose to [batch, seq, head, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        transpose_back = True
    else:
        transpose_back = False
    
    # Get dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # If position_ids is None, use default positions
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device)
    
    # Select the cos and sin values for the positions we need
    # Handle potential size mismatch - ensure cos/sin aren't too long
    if cos.size(0) > position_ids.max() + 1:
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        # If cos/sin are too short, we'll just clip position_ids
        clipped_ids = torch.clamp(position_ids, max=cos.size(0)-1)
        cos = cos[clipped_ids]
        sin = sin[clipped_ids]
    
    # Reshape for broadcasting
    cos = cos.unsqueeze(1).unsqueeze(0)  # [1, seq_len, 1, head_dim]
    sin = sin.unsqueeze(1).unsqueeze(0)  # [1, seq_len, 1, head_dim]
    
    # Apply rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    # Transpose back if needed
    if transpose_back:
        q_embed = q_embed.transpose(1, 2)
        k_embed = k_embed.transpose(1, 2)
    
    return q_embed, k_embed

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """Apply Root Mean Square normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * x * rms

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Apply SwiGLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output after SwiGLU activation
        """
        # SwiGLU activation
        swish = self.w1(x) * torch.sigmoid(self.w2(x) * 1.0)
        x = self.w3(self.dropout(swish))
        return x

class TrueGroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_query_groups=1, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_query_groups = num_query_groups
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        assert num_heads % num_query_groups == 0, "num_heads must be divisible by num_query_groups"
        
        self.heads_per_group = num_heads // num_query_groups
        
        # True GQA: Only project queries for each group (fewer projections)
        # Each query group is shared across multiple attention heads
        self.q_proj = nn.Linear(d_model, self.head_dim * num_query_groups)
        
        # Keys and values are still full-sized (one per head)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Add rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
    
    def forward(self, x, mask=None, is_causal=False):
        batch_size, seq_len, _ = x.shape
        
        # Validate sequence length hasn't exceeded the model's capacity
        if seq_len > self.max_seq_len:
            warnings.warn(f"Input sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}")
            seq_len = min(seq_len, self.max_seq_len)
            x = x[:, :seq_len, :]
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask[:, :seq_len]
                elif mask.dim() == 3:
                    mask = mask[:, :seq_len, :seq_len]
                elif mask.dim() == 4:
                    mask = mask[:, :, :seq_len, :seq_len]
        
        # Validate mask shape if provided
        if mask is not None:
            # Check if mask is a 2D or 4D tensor
            if mask.dim() == 2:
                # Expand mask to 4D: [batch_size, 1, seq_len, seq_len]
                expanded_mask = torch.ones(
                    batch_size, 1, seq_len, seq_len, 
                    device=mask.device, 
                    dtype=mask.dtype
                )
                # Apply token mask to create attention mask
                expanded_mask = expanded_mask * mask.unsqueeze(1).unsqueeze(2)
                mask = expanded_mask
            elif mask.dim() == 3:
                # Expand mask to 4D: [batch_size, 1, seq_len, seq_len]
                mask = mask.unsqueeze(1)
        
        # Create causal mask if needed
        if is_causal:
            # Create a causal mask (lower triangular)
            if mask is None:
                mask = torch.ones(
                    batch_size, 1, seq_len, seq_len, 
                    device=x.device
                )
            
            # Apply causal constraint
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = mask * causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Linear projections
        # For queries, we project to a smaller dimension (num_query_groups * head_dim)
        q = self.q_proj(x)  # [batch_size, seq_len, num_query_groups * head_dim]
        
        # For keys and values, we project to the full dimension
        k = self.k_proj(x)  # [batch_size, seq_len, d_model]
        v = self.v_proj(x)  # [batch_size, seq_len, d_model]
        
        # Reshape queries for grouped query attention
        # [batch_size, seq_len, num_query_groups, head_dim]
        q = q.view(batch_size, seq_len, self.num_query_groups, self.head_dim)
        
        # Reshape keys and values
        # [batch_size, seq_len, num_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Get rotary embeddings
        cos, sin = self.rotary_emb(x, seq_len)
        
        # Apply rotary embeddings to queries and keys
        # First, we need to expand queries to match the number of heads
        # Each query in a group is shared across multiple heads
        q_expanded = torch.repeat_interleave(q, self.heads_per_group, dim=2)  # [batch_size, seq_len, num_heads, head_dim]
        
        # Apply rotary position embeddings
        q_embed, k_embed = apply_rotary_pos_emb(q_expanded, k, cos, sin)
        
        # Transpose for attention matrix multiplication
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q_embed, k_embed.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Apply attention mask
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output

class DeepSeekBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_query_groups=1, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.pre_norm1 = RMSNorm(d_model)
        self.self_attn = TrueGroupedQueryAttention(d_model, num_heads, num_query_groups, dropout, max_seq_len)
        self.pre_norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, is_causal=False):
        # Pre-norm architecture
        residual = x
        x = self.pre_norm1(x)
        x = self.self_attn(x, mask, is_causal)
        x = residual + self.dropout(x)
        
        residual = x
        x = self.pre_norm2(x)
        x = self.ff(x)
        x = residual + self.dropout(x)
        
        return x

class DeepSeekTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, num_query_groups=1, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            DeepSeekBlock(d_model, num_heads, d_ff, num_query_groups, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.max_seq_len = max_seq_len
    
    def forward(self, x, mask=None, is_causal=False):
        # Enforce sequence length constraints
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            warnings.warn(f"Input sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}")
            seq_len = min(seq_len, self.max_seq_len)
            x = x[:, :seq_len, :]
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask[:, :seq_len]
                elif mask.dim() == 3:
                    mask = mask[:, :seq_len, :seq_len]
                elif mask.dim() == 4:
                    mask = mask[:, :, :seq_len, :seq_len]
        
        # Process through layers
        for layer in self.layers:
            x = layer(x, mask, is_causal)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x

# DeepSeek Transformer implementation
class MoEAttention(nn.Module):
    """Mixture of Experts Attention mechanism.
    
    This implements a sparse mixture of experts approach where each token
    is routed to specific experts based on a learned routing function.
    """
    def __init__(self, d_model, num_experts=20, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Calculate head_dim ensuring it's compatible with d_model
        # If d_model is not divisible by num_experts, we'll use the floor division
        # and adjust the expert output dimensions accordingly
        self.head_dim = max(1, d_model // num_experts)
        
        # Each expert is a QKV head
        self.experts = nn.ModuleList([
            nn.Linear(d_model, 3 * self.head_dim) for _ in range(num_experts)
        ])
        
        # Router network to determine which expert to use
        self.router = nn.Linear(d_model, num_experts)
        self.router_dropout = nn.Dropout(0.1)  # Dropout for router logits
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        """Forward pass with expert routing.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Calculate routing probabilities with dropout for regularization
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        router_logits = self.router_dropout(router_logits)
        routing_weights = torch.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # Initialize output tensor
        combined_output = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Get expert weights for this expert
            expert_weights = routing_weights[:, :, i].unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Process through this expert
            qkv = expert(x)  # [batch_size, seq_len, 3 * head_dim]
            qkv = qkv.reshape(batch_size, seq_len, 3, self.head_dim)
            qkv = qkv.permute(2, 0, 1, 3)  # [3, batch_size, seq_len, head_dim]
            
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Scaled dot-product attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, seq_len, seq_len]
            
            # Apply mask if provided
            if mask is not None:
                if mask.dim() == 2:  # [batch_size, seq_len]
                    mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
                
                attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
            
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention weights to values
            expert_output = torch.matmul(attn_weights, v)  # [batch_size, seq_len, head_dim]
            
            # Scale by routing weights
            expert_output = expert_output * expert_weights
            
            # Determine where to place this expert's output in the combined output
            start_idx = min(i * self.head_dim, self.d_model - self.head_dim)
            end_idx = min(start_idx + self.head_dim, self.d_model)
            
            # Add this expert's output to the appropriate slice of the combined output
            combined_output[:, :, start_idx:end_idx] += expert_output
        
        # Final linear projection
        output = self.out_proj(combined_output)
        
        return output

class MoEEncoderBlock(nn.Module):
    """Encoder block using Mixture of Experts attention."""
    def __init__(self, d_model, num_experts, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MoEAttention(d_model, num_experts, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """Forward pass through the MoE encoder block.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Processed tensor
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SLAMv2(nn.Module):
    def __init__(self, d_model, num_experts=20, d_ff=2048, ef_cycles=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.ef_cycles = ef_cycles
        
        # Level 1: Initial Global Attention
        self.initial_block = MoEEncoderBlock(d_model, num_experts, d_ff, dropout)
        
        # Level 2: Encoder Fusion (EF) Cycles
        # 4 parallel encoder blocks for each EF cycle
        self.ef_blocks = nn.ModuleList([
            nn.ModuleList([
                MoEEncoderBlock(d_model, num_experts, d_ff, dropout) 
                for _ in range(4)  # 4 parallel blocks
            ])
            for _ in range(ef_cycles)
        ])
        
        # Level 3: Final Aggregation
        self.final_block = MoEEncoderBlock(d_model, num_experts, d_ff, dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Level 1: Initial Global Attention
        x = self.initial_block(x, mask)
        
        # Level 2: Encoder Fusion (EF) Cycles
        for i in range(self.ef_cycles):
            # Calculate segment boundaries with proper numerical handling
            # Ensure we don't have zero-width segments or out-of-bounds indices
            segment_boundaries = [
                (0, min(int(seq_len * 0.45), seq_len)),                      # Block 1: tokens 0-45%
                (max(int(seq_len * 0.25), 0), min(int(seq_len * 0.65), seq_len)),  # Block 2: tokens 25-65%
                (max(int(seq_len * 0.45), 0), min(int(seq_len * 0.85), seq_len)),  # Block 3: tokens 45-85%
                (max(int(seq_len * 0.65), 0), seq_len)                      # Block 4: tokens 65-100%
            ]
            
            # Add the wraparound segment for Block 4
            if seq_len > 20:  # Only add wraparound if sequence is long enough
                wrap_size = max(min(int(seq_len * 0.05), seq_len // 10), 1)  # At least 1 token, max 10% or seq_len/10
                segment_boundaries[3] = (
                    max(int(seq_len * 0.65), 0),  # Start at 65%
                    seq_len + wrap_size            # End at 100% + 5% wraparound
                )
            
            # Initialize storage for processed segments and token counts (for averaging)
            processed_segments = []
            segment_masks = []
            
            # Process each segment through its corresponding block
            for j, (start, end) in enumerate(segment_boundaries):
                if j == 3 and end > seq_len:  # Handle wraparound for Block 4
                    # Split into two parts: the main segment and the wraparound
                    main_part_end = seq_len
                    wrap_start = 0
                    wrap_end = end - seq_len
                    
                    # Get the main segment
                    main_segment = x[:, start:main_part_end, :]
                    # Get the wraparound segment
                    wrap_segment = x[:, wrap_start:wrap_end, :]
                    # Concatenate them
                    segment = torch.cat([main_segment, wrap_segment], dim=1)
                    
                    # Handle masks for wraparound
                    if mask is not None:
                        if mask.dim() == 2:  # Token mask [batch, seq]
                            # Token masks can be directly concatenated
                            segment_mask = torch.cat([
                                mask[:, start:main_part_end],
                                mask[:, wrap_start:wrap_end]
                            ], dim=1)
                        elif mask.dim() == 3:  # Attention mask [batch, seq, seq]
                            # For attention masks, create a block diagonal mask
                            main_mask = mask[:, start:main_part_end, start:main_part_end]
                            wrap_mask = mask[:, wrap_start:wrap_end, wrap_start:wrap_end]
                            # Create a block diagonal attention mask
                            segment_mask = torch.zeros(
                                batch_size, 
                                main_part_end - start + wrap_end - wrap_start,
                                main_part_end - start + wrap_end - wrap_start,
                                device=x.device
                            )
                            segment_mask[:, :main_part_end-start, :main_part_end-start] = main_mask
                            segment_mask[:, main_part_end-start:, main_part_end-start:] = wrap_mask
                        else:
                            segment_mask = None
                    else:
                        segment_mask = None
                else:
                    # Standard non-wraparound segment
                    real_end = min(end, seq_len)  # Ensure end doesn't exceed seq_len
                    segment = x[:, start:real_end, :]
                    
                    if mask is not None:
                        if mask.dim() == 2:  # Token mask
                            segment_mask = mask[:, start:real_end]
                        elif mask.dim() == 3:  # Attention mask
                            segment_mask = mask[:, start:real_end, start:real_end]
                        else:
                            segment_mask = None
                    else:
                        segment_mask = None
                
                # Process segment through corresponding block
                processed = self.ef_blocks[i][j](segment, segment_mask)
                processed_segments.append((processed, j, start, end))
                segment_masks.append(segment_mask)
            
            # Create new tensor to store the fusion results with token counting for overlap areas
            x_new = torch.zeros_like(x)
            counts = torch.zeros((batch_size, seq_len, 1), device=x.device)
            
            # Add processed segments back to their positions and track counts for averaging
            for processed, j, start, end in processed_segments:
                if j == 3 and end > seq_len:  # Handle Block 4 wraparound
                    main_part_len = seq_len - start
                    # Add main part (from 65% to end)
                    x_new[:, start:seq_len, :] += processed[:, :main_part_len, :]
                    counts[:, start:seq_len, :] += 1
                    
                    # Add wraparound part (from start to 5%)
                    wrap_end = end - seq_len
                    x_new[:, :wrap_end, :] += processed[:, main_part_len:, :]
                    counts[:, :wrap_end, :] += 1
                else:
                    real_end = min(end, seq_len)
                    x_new[:, start:real_end, :] += processed
                    counts[:, start:real_end, :] += 1
            
            # Average the overlapping regions
            # Add a small epsilon to prevent division by zero
            counts = counts.clamp(min=1.0)  # Ensure no zeros in the counts
            x = x_new / counts
        
        # Level 3: Final Aggregation
        x = self.final_block(x, mask)
        
        return x

# DeBERTa implementation
class DisentangledSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_position_embeddings=512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_position_embeddings = max_position_embeddings
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Content projections with explicit bias terms
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Explicit bias terms for mixed-precision training
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        
        # Position projections
        self.pos_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.pos_q_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Register buffers for relative positions - compute once and cache
        self.register_buffer(
            "rel_positions", 
            self._build_relative_positions(max_position_embeddings)
        )
    
    def _build_relative_positions(self, max_len):
        """Compute relative positions matrix based on the maximum length."""
        # Generate relative positions indices
        rel_pos_indices = torch.arange(max_len).unsqueeze(0) - torch.arange(max_len).unsqueeze(1)
        # Shift indices to be non-negative
        rel_pos_indices = rel_pos_indices + max_len - 1
        return rel_pos_indices
    
    def forward(self, x, rel_pos_emb, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Get relative position embeddings for the sequence length
        rel_pos = self.rel_positions[:seq_len, :seq_len]
        # Gather the relative position embeddings for each position
        rel_pos_emb_slice = rel_pos_emb[rel_pos]  # Shape: [seq_len, seq_len, d_model]
        
        # Content projections with explicit bias terms
        q_content = self.q_proj(x) + self.q_bias  # Add separate bias
        k_content = self.k_proj(x)  # Use default bias in Linear
        v = self.v_proj(x) + self.v_bias  # Add separate bias
        
        # Project position embeddings
        rel_pos_k = self.pos_k_proj(rel_pos_emb_slice)  # [seq_len, seq_len, d_model]
        rel_pos_q = self.pos_q_proj(rel_pos_emb_slice)  # [seq_len, seq_len, d_model]
        
        # Reshape for attention computation
        q_content = q_content.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_content = k_content.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Reshape for relative position
        rel_pos_k = rel_pos_k.view(seq_len, seq_len, self.num_heads, self.head_dim)
        rel_pos_q = rel_pos_q.view(seq_len, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q_content = q_content.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        k_content = k_content.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Content-to-content (c2c) attention
        attn_scores_c2c = torch.matmul(q_content, k_content.transpose(-2, -1)) * self.scale
        
        # Compute content-to-position (c2p) and position-to-content (p2c) attention more efficiently
        # Instead of the loops, reshape and use batch matrix multiplication
        
        # Content-to-position (c2p) attention
        # Reshape rel_pos_k to [seq_len, num_heads, seq_len, head_dim] and transpose properly
        rel_pos_k = rel_pos_k.permute(0, 2, 1, 3)  # [seq_len, num_heads, seq_len, head_dim]
        # Reshape q_content for broadcasting with rel_pos_k
        q_content_expanded = q_content.unsqueeze(2)  # [batch_size, num_heads, 1, seq_len, head_dim]
        rel_pos_k_expanded = rel_pos_k.unsqueeze(0)  # [1, seq_len, num_heads, seq_len, head_dim]
        
        # Compute c2p attention scores for all positions at once
        # [batch_size, seq_len, num_heads, seq_len, head_dim] * [batch_size, seq_len, num_heads, head_dim, seq_len]
        attn_scores_c2p = torch.zeros_like(attn_scores_c2c)
        for i in range(seq_len):
            attn_scores_c2p[:, :, i, :] = torch.matmul(
                q_content[:, :, i:i+1, :], 
                rel_pos_k[i].transpose(-2, -1)
            ).squeeze(2) * self.scale
        
        # Position-to-content (p2c) attention
        rel_pos_q = rel_pos_q.permute(1, 2, 0, 3)  # [seq_len, num_heads, seq_len, head_dim]
        # Compute p2c attention scores for all positions at once
        attn_scores_p2c = torch.zeros_like(attn_scores_c2c)
        for i in range(seq_len):
            attn_scores_p2c[:, :, :, i] = torch.matmul(
                rel_pos_q[i], 
                k_content.transpose(-2, -1)[:, :, :, i:i+1]
            ).squeeze(-1) * self.scale
        
        # Combine attention scores (DeBERTa paper allows omitting position-to-position)
        attn_scores = attn_scores_c2c + attn_scores_c2p + attn_scores_p2c
        
        # Handle the mask properly
        if mask is not None:
            # Expand mask to 4D if needed
            if mask.dim() == 2:
                # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
                mask = mask.unsqueeze(1)
            
            # Apply mask
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)  # Use -1e10 for numerical stability
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output

class DeBERTaEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, max_position_embeddings=512):
        super().__init__()
        self.self_attn = DisentangledSelfAttention(d_model, num_heads, dropout, max_position_embeddings)
        # Use RMSNorm instead of LayerNorm for better performance
        self.norm1 = RMSNorm(d_model)
        # Use SwiGLU instead of standard FeedForward for better performance
        self.ff = SwiGLU(d_model, d_ff, dropout)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, rel_pos_emb, mask=None):
        # Pre-norm architecture (like DeepSeek) instead of post-norm
        residual = x
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, rel_pos_emb, mask)
        x = residual + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm
        residual = x
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = residual + self.dropout(ff_output)
        
        return x

class DeBERTaEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_position_embeddings=512):
        super().__init__()
        self.d_model = d_model
        self.max_position_embeddings = max_position_embeddings
        
        # Relative position embeddings table (2*max_len - 1) positions
        self.pos_embeddings = nn.Parameter(
            torch.zeros(2 * max_position_embeddings - 1, d_model)
        )
        nn.init.normal_(self.pos_embeddings, mean=0, std=0.02)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            DeBERTaEncoderBlock(d_model, num_heads, d_ff, dropout, max_position_embeddings)
            for _ in range(num_layers)
        ])
        
        # Final normalization - use RMSNorm for consistency
        self.norm = RMSNorm(d_model)
        
        # Optional: Enhanced Mask Decoder (for masked language modeling tasks)
        # Excluded in this implementation as it focuses on classification tasks
    
    def forward(self, x, mask=None):
        """
        Forward pass through the DeBERTa encoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask of shape [batch_size, seq_len] or [batch_size, seq_len, seq_len]
                  where 1 indicates tokens to attend to and 0 indicates tokens to ignore
                  
        Returns:
            Encoded representation of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Check mask is properly formatted
        if mask is not None and mask.dim() == 2:
            # Convert [batch_size, seq_len] mask to attention mask [batch_size, seq_len, seq_len]
            # Where each token can attend to all other non-masked tokens
            attn_mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            # Apply causal masking if required (for autoregressive models)
            # attn_mask = attn_mask & torch.tril(torch.ones_like(attn_mask))
        else:
            attn_mask = mask
        
        # Process through encoder layers
        for layer in self.layers:
            x = layer(x, self.pos_embeddings, attn_mask)
        
        # Final layer normalization
        x = self.norm(x)
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train a model and evaluate it on validation data.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train for
        device: Device to train on (cpu or cuda)
        
    Returns:
        train_losses: List of training losses for each epoch
        val_losses: List of validation losses for each epoch
        train_accs: List of training accuracies for each epoch
        val_accs: List of validation accuracies for each epoch
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize tracking lists
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    for epoch in range(num_epochs):
        try:
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Track statistics
                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                except Exception as e:
                    print(f"Error in training batch: {e}")
                    continue
            
            # Calculate epoch statistics
            if train_total > 0:  # Avoid division by zero
                train_loss = train_loss / train_total
                train_acc = 100.0 * train_correct / train_total
            else:
                train_loss = float('nan')
                train_acc = 0.0
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    try:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        # Track statistics
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue
            
            # Calculate epoch statistics
            if val_total > 0:  # Avoid division by zero
                val_loss = val_loss / val_total
                val_acc = 100.0 * val_correct / val_total
            else:
                val_loss = float('nan')
                val_acc = 0.0
            
            # Store results
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Print epoch results
            print(f'Epoch {epoch+1}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.2f}%')
            
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            # Add nan values for this epoch to maintain list length
            train_losses.append(float('nan'))
            val_losses.append(float('nan'))
            train_accs.append(0.0)
            val_accs.append(0.0)
    
    return train_losses, val_losses, train_accs, val_accs

def main():
    """Main function to run the transformer architecture comparison experiment."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_classes = 5
    seq_len = 50
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Standard Transformer parameters
    std_num_layers = 4
    
    # SLAM parameters
    ef_cycles = 3
    
    # DeepSeek parameters
    deepseek_num_layers = 4
    deepseek_num_query_groups = 2  # Group query attention parameter
    
    # DeBERTa parameters
    deberta_num_layers = 4
    max_position_embeddings = 512
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    num_train_samples = 1000
    num_val_samples = 200
    
    X_train, y_train = generate_synthetic_data(num_train_samples, seq_len, d_model, num_classes)
    X_val, y_val = generate_synthetic_data(num_val_samples, seq_len, d_model, num_classes)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize models
    print("Initializing models...")
    
    # Standard Transformer
    std_encoder = StandardTransformer(d_model, num_heads, d_ff, std_num_layers)
    std_model = TransformerClassifier(std_encoder, d_model, num_classes, seq_len)
    
    # SLAM v1
    slam_v1_encoder = SLAMv1(d_model, num_heads, d_ff, ef_cycles)
    slam_v1_model = TransformerClassifier(slam_v1_encoder, d_model, num_classes, seq_len)
    
    # SLAM v2
    slam_v2_encoder = SLAMv2(d_model, num_experts=num_heads*2, d_ff=d_ff, ef_cycles=ef_cycles)
    slam_v2_model = TransformerClassifier(slam_v2_encoder, d_model, num_classes, seq_len)
    
    # DeepSeek Transformer
    deepseek_encoder = DeepSeekTransformer(
        d_model, 
        num_heads, 
        d_ff, 
        deepseek_num_layers, 
        deepseek_num_query_groups,
        max_seq_len=seq_len
    )
    deepseek_model = TransformerClassifier(deepseek_encoder, d_model, num_classes, seq_len)
    
    # DeBERTa Encoder
    deberta_encoder = DeBERTaEncoder(
        d_model, 
        num_heads, 
        d_ff, 
        deberta_num_layers, 
        max_position_embeddings=max_position_embeddings
    )
    deberta_model = TransformerClassifier(deberta_encoder, d_model, num_classes, seq_len)
    
    # Count parameters - do this before moving to device
    std_params = sum(p.numel() for p in std_model.parameters())
    slam_v1_params = sum(p.numel() for p in slam_v1_model.parameters())
    slam_v2_params = sum(p.numel() for p in slam_v2_model.parameters())
    deepseek_params = sum(p.numel() for p in deepseek_model.parameters())
    deberta_params = sum(p.numel() for p in deberta_model.parameters())
    
    # Loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    std_optimizer = optim.Adam(std_model.parameters(), lr=learning_rate)
    slam_v1_optimizer = optim.Adam(slam_v1_model.parameters(), lr=learning_rate)
    slam_v2_optimizer = optim.Adam(slam_v2_model.parameters(), lr=learning_rate)
    deepseek_optimizer = optim.Adam(deepseek_model.parameters(), lr=learning_rate)
    deberta_optimizer = optim.Adam(deberta_model.parameters(), lr=learning_rate)
    
    # Define a dictionary of models for easier iteration
    models = {
        'Standard': (std_model, std_optimizer),
        'SLAM v1': (slam_v1_model, slam_v1_optimizer),
        'SLAM v2': (slam_v2_model, slam_v2_optimizer),
        'DeepSeek': (deepseek_model, deepseek_optimizer),
        'DeBERTa': (deberta_model, deberta_optimizer),
    }
    
    # Dict to store results
    results = {}
    
    # Train models
    for model_name, (model, optimizer) in models.items():
        print(f"\nTraining {model_name} Transformer...")
        start_time = time.time()
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs, device
        )
        training_time = time.time() - start_time
        print(f"{model_name} training time: {training_time:.2f} seconds")
        
        results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'time': training_time
        }
    
    # Plot results
    epochs = range(1, num_epochs + 1)
    
    # Plot training and validation loss
    plt.figure(figsize=(20, 15))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    for model_name, res in results.items():
        plt.plot(epochs, res['train_losses'], '-', label=f'{model_name} Train')
        plt.plot(epochs, res['val_losses'], '--', label=f'{model_name} Val')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    for model_name, res in results.items():
        plt.plot(epochs, res['train_accs'], '-', label=f'{model_name} Train')
        plt.plot(epochs, res['val_accs'], '--', label=f'{model_name} Val')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Validation accuracy comparison
    plt.subplot(2, 2, 3)
    for model_name, res in results.items():
        plt.plot(epochs, res['val_accs'], '-', label=model_name)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Training time comparison
    plt.subplot(2, 2, 4)
    model_names = list(results.keys())
    times = [results[name]['time'] for name in model_names]
    plt.bar(model_names, times)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('transformer_comparison.png')
    
    # Print final results
    print("\nFinal Results:")
    for model_name, res in results.items():
        print(f"{model_name} - Val Accuracy: {res['val_accs'][-1]:.2f}%, Training Time: {res['time']:.2f}s")
    
    # Parameter counts
    print(f"\nModel Parameters:")
    param_counts = {
        'Standard': std_params,
        'SLAM v1': slam_v1_params,
        'SLAM v2': slam_v2_params,
        'DeepSeek': deepseek_params,
        'DeBERTa': deberta_params
    }
    
    for model_name, count in param_counts.items():
        print(f"{model_name} Transformer: {count:,}")
    
    # Save a summary to file
    with open('transformer_comparison_results.txt', 'w') as f:
        f.write("Transformer Architecture Comparison Results\n")
        f.write("=========================================\n\n")
        
        f.write("Model Parameters:\n")
        for model_name, count in param_counts.items():
            f.write(f"{model_name} Transformer: {count:,}\n")
        
        f.write("\nFinal Validation Accuracies:\n")
        for model_name, res in results.items():
            f.write(f"{model_name} Transformer: {res['val_accs'][-1]:.2f}%\n")
        
        f.write("\nTraining Times:\n")
        for model_name, res in results.items():
            f.write(f"{model_name} Transformer: {res['time']:.2f}s\n")

if __name__ == "__main__":
    main()