import torch
import torch.nn as nn
import warnings
from typing import Optional, Tuple, List, Union, Dict

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

# Example usage
if __name__ == "__main__":
    # Example parameters
    d_model = 64
    num_experts = 20
    d_ff = 256
    ef_cycles = 3
    seq_len = 50
    batch_size = 8
    num_classes = 5
    
    # Create a sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize SLAM v2 model
    slam_v2 = SLAMv2(d_model, num_experts, d_ff, ef_cycles)
    
    # Create classifier wrapper
    classifier = TransformerClassifier(slam_v2, d_model, num_classes, seq_len)
    
    # Test with no mask
    print("Testing with no mask...")
    output = slam_v2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test classifier
    logits = classifier(x)
    print(f"Classifier output shape: {logits.shape}")
    
    # Test with token mask (2D)
    print("\nTesting with token mask (2D)...")
    token_mask = torch.ones(batch_size, seq_len)
    token_mask[:, seq_len//2:] = 0  # Mask out second half
    output_token_mask = slam_v2(x, token_mask)
    print(f"Token mask shape: {token_mask.shape}")
    print(f"Output with token mask shape: {output_token_mask.shape}")
    
    # Test with attention mask (3D)
    print("\nTesting with attention mask (3D)...")
    attn_mask = torch.ones(batch_size, seq_len, seq_len)
    attn_mask[:, :, seq_len//2:] = 0  # Mask out attention to second half
    output_attn_mask = slam_v2(x, attn_mask)
    print(f"Attention mask shape: {attn_mask.shape}")
    print(f"Output with attention mask shape: {output_attn_mask.shape}")
    
    print("\nSLAM v2 architecture test complete!")