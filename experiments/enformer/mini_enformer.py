import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import Optional, Tuple, Dict, List

def exponential_linspace_int(start: int, end: int, num: int, divisible_by: int = 1) -> List[int]:
    """reference lines 328-334 enformer.py"""
    def round_to_divisible(x):
        return int(np.round(x / divisible_by) * divisible_by)
        
    if num == 1:
        return [end]
        
    base = np.exp(np.log(end / start) / (num - 1))
    return [round_to_divisible(start * base**i) for i in range(num)]


class RelativePositionalBias(nn.Module):
    #max_distance = 1000 covers +/- 1 kb genomic context
    # num_bases = 16 covers 16 basis functions as in Enformer
    # min_half_life = 3.0
    #max_time = 10000.0 
    def __init__(self, num_heads: int, max_distance: int = 1000,
                 num_bases: int = 16, feature_type: str = 'exponential',
                 min_half_life: float = 3.0, max_time: float = 10000.0):
      
        super(RelativePositionalBias, self).__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.num_bases = num_bases
        self.feature_type = feature_type
        self.min_half_life = min_half_life
        self.max_time = max_time

        # precompute relative positions [-max_distance, +max_distance]
        positions = torch.arange(-max_distance, max_distance + 1) # creates tensor of all possible relative positions between query and key
        self.register_buffer('positions', positions)

        # precompute basis features for all relative positions
        basis_features = self._compute_basis_features(positions)
        self.register_buffer('basis_features', basis_features) # store in buffer to avoid recomputing

        # learnable weights for basis functions
        self.basis_weights = nn.Parameter(
            torch.randn(num_heads, basis_features.size(-1))
        )

    def _compute_basis_features(self, positions: torch.Tensor) -> torch.Tensor:

        abs_pos = positions.abs().float()

        if self.feature_type == 'exponential': #for TF motifs
            return self._exponential_basis(abs_pos)
        elif self.feature_type == 'gamma': # for mid-range interactions
            return self._gamma_basis(abs_pos)
        elif self.feature_type == 'central_mask': # for local regions
            return self._central_mask(abs_pos)
        elif self.feature_type == 'cosine': # was in Enformer repo
            return self._cosine_basis(positions)
        elif self.feature_type == 'linear_masks': # was in Enformer repo
            return self._linear_masks(abs_pos)
        elif self.feature_type == 'sin_cos': # was in Enformer repo
            return self._sin_cos_basis(positions)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def _exponential_basis(self, abs_pos: torch.Tensor) -> torch.Tensor: # input is a 1D tensor of absolute relative positions
        max_range = math.log2(self.max_distance)
        half_lives = torch.pow(
            2.0, torch.linspace(self.min_half_life, max_range, self.num_bases)
        ).view(1, -1)  # Shape: (1, num_bases)
        features = torch.exp(-math.log(2.0) * abs_pos[:, None] / half_lives)
        return features  # Shape: [2*max_distance+1, num_bases], tensor of exponential decay features

    def _gamma_basis(self, abs_pos: torch.Tensor) -> torch.Tensor: # input is a 1D tensor of absolute relative positions
        seq_length = float(self.max_distance)
        stddev = seq_length / (2 * self.num_bases) #fixed standard deviation for all basis functions
        start_mean = seq_length / self.num_bases #where first gamma bump will be centered
        mean = torch.linspace(start_mean, seq_length, steps=self.num_bases).view(1, -1)
        concentration = (mean / stddev) ** 2 # computes shape parameter for gamma distribution (k)
        rate = mean / (stddev ** 2) # computes rate parameter (Beta) for gamma distribution
        x = abs_pos[:, None] 
        log_unnormalized_prob = (concentration - 1) * torch.log(x + 1e-8) - rate * x # log unnormalized probability density function
        log_normalization = (
            torch.lgamma(concentration) - concentration * torch.log(rate + 1e-8) # normalization constant
        )
        probs = torch.exp(log_unnormalized_prob - log_normalization) # gets actual gamma pdf
        probs = probs / probs.max(dim=1, keepdim=True).values  # Normalize
        probs += 1e-8
        return probs  # Shape: [2*max_distance+1, num_bases]

    def _central_mask(self, abs_pos: torch.Tensor) -> torch.Tensor:
        center_widths = torch.pow(2.0, torch.arange(1, self.num_bases + 1)).view(1, -1) #from repo, defines central region like [2, 4, 8, 16, etc,...]
        features = (center_widths > abs_pos[:, None]).float() # applies mask
        return features  # Shape: [2*max_distance+1, num_bases]

    def _cosine_basis(self, positions: torch.Tensor) -> torch.Tensor:
        """cosine positional features"""
        periodicity = 1.25 * torch.pow(2.0, torch.arange(self.num_bases)).view(1, -1)
        features = torch.cos(2 * math.pi * positions[:, None] / periodicity)
        return features  # Shape: [2*max_distance+1, num_bases] - value is 1 if within basis region else 0

    def _linear_masks(self, abs_pos: torch.Tensor) -> torch.Tensor:
        """each mask will only focus on one specific relative position
           e.g. Basis 0: only look at position 0, Basis 1: only look at position 1, etc."""
        distances = torch.arange(0, self.num_bases).view(1, -1) #from repo, defines central region like [0, 1, 2, 3]
        features = (distances == abs_pos[:, None]).float() # applies mask
        return features  # Shape: [2*max_distance+1, num_bases]

    def _sin_cos_basis(self, positions: torch.Tensor) -> torch.Tensor:
        """pretty much the same as Attention is all you need. Sin for even positions, Cos for odd positions"""
        if self.num_bases % 2 != 0:
            raise ValueError("num_bases must be even for sin/cos features.")
        i = torch.arange(0, self.num_bases, 2).float().view(1, -1)
        div_term = torch.pow(self.max_time, i / self.num_bases)
        pos_enc = torch.cat([
            torch.sin(positions[:, None] / div_term),
            torch.cos(positions[:, None] / div_term)
        ], dim=-1)
        return pos_enc  # Shape: [2*max_distance+1, num_bases]

    def forward(self, query_len: int, key_len: int) -> torch.Tensor:
        """Return relative positional bias tensor.

        Args:
            query_len (int): Length of query sequence.
            key_len (int): Length of key sequence.

        Returns:
            torch.Tensor: Bias of shape [num_heads, query_len, key_len]
        """
        # compute relative positions between query and key
        relative_positions = torch.arange(key_len) - torch.arange(query_len)[:, None]
        relative_positions = relative_positions.clamp(-self.max_distance, self.max_distance) # clips max distance i.e. long distances are just treated as max distance
        relative_positions += self.max_distance  # Shift to index [0, 2*max_distance]

        # look up precomputed basis features
        basis = self.basis_features[relative_positions]  # Shape: [query_len, key_len, num_bases]

        # combine with learnable weights
        bias = torch.einsum('qkb,hb->hqk', basis, self.basis_weights) #for each head, compute weighted sum of basis features
        return bias  # Shape: [num_heads, query_len, key_len]


class GELU(nn.Module):
    """GELU activation function
       Each convolutional block in Enformer uses GELU activation after batch norm
       each FFN in transformer block uses GELU activation after linear layer
       """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(1.702 * x) * x
    

class ConvBlock(nn.Module):
    """BatchNorm -> GELU -> Conv1D see line 88-98 enformer.py"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, 
                 padding: int = 0, dropout: float = 0.0):
        super(ConvBlock, self).__init__()
        
        self.bn = nn.BatchNorm1d(in_channels)
        self.activation = GELU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x

class SoftmaxPooling1D(nn.Module):
    """see enformer.py 247-284"""
    def __init__(self, pool_size: int = 2, per_channel: bool = True, w_init_scale: float = 2.0):
        super().__init__()
        self.pool_size = pool_size
        self.per_channel = per_channel
        self.w_init_scale = w_init_scale
        self.logit_linear = None
    
    def _initialize(self, num_features: int, device: torch.device):
        if self.logit_linear is None:
            output_size = num_features if self.per_channel else 1
            self.logit_linear = nn.Linear(num_features, output_size, bias=False) # linear layer to compute weights for pooling
            
            
            with torch.no_grad():
                if self.per_channel:
                    self.logit_linear.weight.data = torch.eye(num_features) * self.w_init_scale # each channel gets its own weight
                else:
                    self.logit_linear.weight.data.fill_(self.w_init_scale)
            
            self.logit_linear = self.logit_linear.to(device)
            self.add_module('logit_linear', self.logit_linear)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_features, seq_len = x.shape
        
        # make sequence divisible by pool_size
        if seq_len % self.pool_size != 0:
            trim_len = seq_len - (seq_len % self.pool_size)
            x = x[:, :, :trim_len]
            seq_len = trim_len
            print(f"Warning: Input sequence length {seq_len} is not divisible by pool_size {self.pool_size}. Trimming to {trim_len}.")
        
        self._initialize(num_features, x.device)
        
        # reshape for pooling: (batch, features, seq_len//pool_size, pool_size) every 2 elements go together
        x_reshaped = x.view(batch_size, num_features, seq_len // self.pool_size, self.pool_size)
        
        # transpose for linear layer: (batch, seq_len//pool_size, pool_size, features)
        x_transposed = x_reshaped.permute(0, 2, 3, 1)
        
        # apply linear transformation 
        logits = self.logit_linear(x_transposed)
        
        # softmax over pool_size dimension
        weights = F.softmax(logits, dim=-2)  
        
        # Apply weights and sum over pool_size
        if self.per_channel:
            weighted = x_transposed * weights
        else:
            weighted = x_transposed * weights  # broadcasts automatically
        
        # sum over pool_size dimension and transpose back
        pooled = weighted.sum(dim=-2).permute(0, 2, 1)
        
        return pooled

class ResidualBlock(nn.Module):
    """wrapper for residual block"""
    def __init__(self, module: nn.Module):
        super(ResidualBlock, self).__init__()
        self.module = module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)

class MultiHeadAttention(nn.Module):
    """104-208 in enformer.py"""
    def __init__(self, d_model: int, num_heads: int, 
                 key_size: int = 64, value_size: int = None,
                 attention_dropout: float = 0.05, output_dropout: float = 0.4):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or d_model // num_heads
        
        # linear projections
        self.w_q = nn.Linear(d_model, num_heads * key_size, bias=False)
        self.w_k = nn.Linear(d_model, num_heads * key_size, bias=False)
        self.w_v = nn.Linear(d_model, num_heads * self.value_size, bias=False)
        self.w_o = nn.Linear(num_heads * self.value_size, d_model, bias=False)
        
        # relative positional bias
        self.relative_bias = RelativePositionalBias(num_heads)
        
        # dropouts
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(output_dropout)
        
        # zero initialize output projection 
        with torch.no_grad():
            self.w_o.weight.zero_()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # linear projections and reshape
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.value_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.key_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # add relative positional bias
        relative_bias = self.relative_bias(seq_len, seq_len)
        scores = scores + relative_bias.unsqueeze(0)
        
        # apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # apply attention to values
        out = torch.matmul(attention_weights, v)
        
        # concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.value_size)
        out = self.w_o(out)
        out = self.output_dropout(out)
        
        return out, attention_weights  #  return both output and attention weights

class TransformerBlock(nn.Module):
    """66-101 in enformer.py"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.4):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            key_size=64,
            value_size=d_model // num_heads,
            attention_dropout=0.05,
            output_dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        attn_out, attention_weights = self.attention(self.norm1(x))  
        x = x + attn_out
        
        
        x = x + self.ffn(self.norm2(x))
        
        return x, attention_weights  

class TargetLengthCrop1D(nn.Module):
    def __init__(self, target_length=125):
        super().__init__()
        self.target_length = target_length
    def forward(self, x):
        seq_len = x.shape[2]
        trim = (seq_len - self.target_length) // 2
        return x[:, :, trim:trim+self.target_length]

class NetDeepHistoneEnformer(nn.Module):
    def __init__(self, 
                 input_channels: int = 5,
                 channels: int = 1536,
                 num_transformer_layers: int = 11,
                 num_heads: int = 8,
                 dropout: float = 0.4,
                 num_histones: int = 7,
                 pooling_type: str = 'attention'): 
        super(NetDeepHistoneEnformer, self).__init__()
        
        print(f'DeepHistone-Enformer Hybrid (1kb) initialized')
        print(f'Channels: {channels}, Transformer layers: {num_transformer_layers}, Heads: {num_heads}')
        
        self.channels = channels
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.pooling_type = pooling_type

        stem_out_channels = channels // 2
        #print(f"[DEBUG INIT] Target stem_out_channels: {stem_out_channels}")

        self.stem = self._make_stem(input_channels, stem_out_channels, pooling_type)
        
        # DEBUG: Get actual stem output channels
        actual_stem_out = self._get_stem_output_channels()
        #print(f"[DEBUG INIT] Actual stem output channels: {actual_stem_out}")
        
        # convolutional tower (exponentially increasing channels)
        self.conv_tower = self._build_conv_tower(actual_stem_out, channels, pooling_type, num_blocks=2) #choosing 2 blocks for my 1000 bp sequence for now
        
        # transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(channels, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # final processing
        self.final_conv = ConvBlock(channels, channels * 2, kernel_size=1, dropout=dropout / 8)
        
        # classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, num_histones),
            nn.Sigmoid()
        )
        
        # store attention weights for visualization
        self.attention_weights = []

    def _get_stem_output_channels(self):
        """get actual output channels from stem"""
        # Find the last Conv1d or ConvBlock in stem
        for module in reversed(list(self.stem.modules())):
            if isinstance(module, nn.Conv1d):
                return module.out_channels
            elif isinstance(module, ConvBlock):
                return module.conv.out_channels
        return None
    
    #creates initial processing layer
    def _make_stem(self, input_channels: int, stem_out_channels: int, pooling_type: str) -> nn.Sequential:
        """
        Build the stem: initial convolution + residual + pooling.
        """
        #print(f"[DEBUG STEM] Building stem: {input_channels} -> {stem_out_channels}")
        layers = []

        # Initial wide kernel convolution (motif detector)
        layers.append(nn.Conv1d(input_channels, stem_out_channels, kernel_size=15, padding=7))
        #print(f"[DEBUG STEM] Added initial conv: {input_channels} -> {stem_out_channels}")

        # Residual block for early refinement
        layers.append(ResidualBlock(
            ConvBlock(stem_out_channels, stem_out_channels, kernel_size=1)
        ))
        #print(f"[DEBUG STEM] Added residual block: {stem_out_channels} -> {stem_out_channels}")

        # pooling: reduce sequence length early
        if pooling_type == 'attention':
            layers.append(SoftmaxPooling1D(pool_size=2, w_init_scale=2.0))
        else:
            layers.append(nn.MaxPool1d(kernel_size=2, padding=0))
        #print(f"[DEBUG STEM] Added pooling layer")

        return nn.Sequential(*layers)
    

    def _build_conv_tower(self, in_channels: int, out_channels: int, pooling_type: str, num_blocks: int = 2) -> nn.Sequential:
        """Build conv tower with 2 blocks for 1000bp sequences"""
        
        # Use 2 blocks instead of 6 (need something that divides 1000 so we go 1000 -> 500 -> 250)
        filter_list = exponential_linspace_int(
            start=in_channels, 
            end=out_channels, 
            num=num_blocks,  # Use the parameter (2 by default)
            divisible_by=128
        )
        
        blocks = []
        current_channels = in_channels
        
        for i, num_filters in enumerate(filter_list):
            block = nn.Sequential(
                ConvBlock(current_channels, num_filters, kernel_size=5, padding=2),
                ResidualBlock(ConvBlock(num_filters, num_filters, kernel_size=1)),
                SoftmaxPooling1D(pool_size=2, per_channel=True, w_init_scale=2.0) if pooling_type == 'attention' 
                else nn.MaxPool1d(kernel_size=2, padding=0)
            )
            blocks.append(block)
            current_channels = num_filters
        
        return nn.Sequential(*blocks)


    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(f"[DEBUG FORWARD] Input shape: {x.shape}")
        
        
        # stem conv -> res -> softmax
        x = self.stem(x)
        #print(f"[DEBUG FORWARD] After stem: {x.shape}")
        
        # convolutional tower
        for i, block in enumerate(self.conv_tower):
            #print(f"[DEBUG FORWARD] Before conv_tower block {i}: {x.shape}")
            x = block(x)
           # print(f"[DEBUG FORWARD] After conv_tower block {i}: {x.shape}")
        
        # Convert back to (batch_size, seq_len, channels) for transformer
        x = x.transpose(1, 2)
        #print(f"[DEBUG FORWARD] After transpose for transformer: {x.shape}")
        
        # Transformer encoder
        self.attention_weights = []
        for i, transformer_block in enumerate(self.transformer):
           # print(f"[DEBUG FORWARD] Before transformer block {i}: {x.shape}")
            x, attn_weights = transformer_block(x)
            self.attention_weights.append(attn_weights)
           # print(f"[DEBUG FORWARD] After transformer block {i}: {x.shape}")
        
        # final processing
        x = x.transpose(1, 2)  # Back to (batch_size, channels, seq_len)
       # print(f"[DEBUG FORWARD] After transpose back: {x.shape}")
        x = self.final_conv(x)
       # print(f"[DEBUG FORWARD] After final_conv: {x.shape}")
        
        # classification
        output = self.classifier(x)
      #  print(f"[DEBUG FORWARD] Final output: {output.shape}")
        
        return output
    
    # don't really use these functions
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Return stored attention weights"""
        return self.attention_weights

    def get_contribution_scores(self, x: torch.Tensor, target_class: int = 0) -> torch.Tensor:
        """Get contribution scores using gradients"""
        x.requires_grad_(True)
        
        # forward pass
        output = self.forward(x)
        
        # get gradients for the target class
        target_output = output[:, target_class].sum()
        target_output.backward()
        
        # return input gradients as contribution scores
        return x.grad.abs().sum(dim=1)  # sum over channel dimension

        
    

class DeepHistoneEnformer:
    """Training wrapper"""
    def __init__(self, 
                 use_gpu: bool = True,
                 learning_rate: float = 0.001,
                 input_channels: int = 5,
                 channels: int = 768,
                 num_transformer_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.4,
                 num_histones: int = 7,
                 pooling_type: str = 'attention'):
        
        #instance of enformer
        self.forward_fn = NetDeepHistoneEnformer(
            input_channels=input_channels,
            channels=channels,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_histones=num_histones,
            pooling_type=pooling_type
        )
        
        #loss function and adam optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.forward_fn.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # moves to gpu if available
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.forward_fn = self.forward_fn.cuda()
            self.criterion = self.criterion.cuda()
    
    # keeping the same learning rate update as in deephistone
    def updateLR(self, fold: float):
        """Update learning rate by multiplication factor"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= fold
    
    # concatenating the inputs
    def _combine_inputs(self, seq_batch: np.ndarray, dns_batch: np.ndarray) -> torch.Tensor:
        """Combine DNA sequence and DNase into 5-channel input"""
        # seq_batch: (batch_size, 1000, 4) - one-hot encoded DNA
        # dns_batch: (batch_size, 1000, 1) - DNase signal
        
        # Debug prints
        # print("="*50)
        # print("DEBUG: _combine_inputs called")
        # print(f"seq_batch type: {type(seq_batch)}")
        # print(f"seq_batch shape: {seq_batch.shape}")
        # print(f"seq_batch dtype: {seq_batch.dtype}")
        # print(f"dns_batch type: {type(dns_batch)}")
        # print(f"dns_batch shape: {dns_batch.shape}")
        # print(f"dns_batch dtype: {dns_batch.dtype}")

        seq_tensor = torch.tensor(seq_batch, dtype=torch.float32)
        dns_tensor = torch.tensor(dns_batch, dtype=torch.float32)
        
        # print(f"seq_tensor shape: {seq_tensor.shape}")
        # print(f"dns_tensor shape: {dns_tensor.shape}")
        # ensure DNase has the right shape
        if dns_tensor.dim() == 2:
            dns_tensor = dns_tensor.unsqueeze(-1)

        seq_tensor = seq_tensor.transpose(1, 2)  # (16, 1000, 4)
        dns_tensor = dns_tensor.transpose(1, 2)  # (16, 1000, 1)
        
        # concatenate to create 5-channel input
        combined = torch.cat([seq_tensor, dns_tensor], dim=-1) 

        combined = combined.permute(0, 2, 1)
        
        return combined
    
    def train_on_batch(self, seq_batch: np.ndarray, dns_batch: np.ndarray, lab_batch: np.ndarray) -> float:
        """Train on a single batch"""
        self.forward_fn.train()
        
        # combine inputs and convert to tensors
        combined_input = self._combine_inputs(seq_batch, dns_batch)
        lab_batch = torch.tensor(lab_batch, dtype=torch.float32)
        
        if self.use_gpu:
            combined_input = combined_input.cuda()
            lab_batch = lab_batch.cuda()
        
        # forward pass
        output = self.forward_fn(combined_input)
        loss = self.criterion(output, lab_batch)
        
        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.cpu().item()
    
    def eval_on_batch(self, seq_batch: np.ndarray, dns_batch: np.ndarray, lab_batch: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate on a single batch"""
        self.forward_fn.eval()
        
        with torch.no_grad():
            # combine inputs and convert to tensors
            combined_input = self._combine_inputs(seq_batch, dns_batch)
            lab_batch = torch.tensor(lab_batch, dtype=torch.float32)
            
            if self.use_gpu:
                combined_input = combined_input.cuda()
                lab_batch = lab_batch.cuda()
            
            # forward pass
            output = self.forward_fn(combined_input)
            loss = self.criterion(output, lab_batch)
            
            return loss.cpu().item(), output.cpu().numpy()
    
    def test_on_batch(self, seq_batch: np.ndarray, dns_batch: np.ndarray) -> np.ndarray:
        """Test on a single batch (no labels)"""
        self.forward_fn.eval()
        
        with torch.no_grad():
            # combine inputs and convert to tensors
            combined_input = self._combine_inputs(seq_batch, dns_batch)
            
            if self.use_gpu:
                combined_input = combined_input.cuda()
            
            # forward pass
            output = self.forward_fn(combined_input)
            
            return output.cpu().numpy()
    
    # again 
    def get_attention_weights(self, seq_batch: np.ndarray, dns_batch: np.ndarray) -> List[torch.Tensor]:
        """attention weights for visualization"""
        self.forward_fn.eval()
        
        with torch.no_grad():
            combined_input = self._combine_inputs(seq_batch, dns_batch)
            
            if self.use_gpu:
                combined_input = combined_input.cuda()
            
            # forward pass to compute attention weights
            _ = self.forward_fn(combined_input)
            
            return self.forward_fn.get_attention_weights()
    
    def get_contribution_scores(self, seq_batch: np.ndarray, dns_batch: np.ndarray, target_class: int = 0) -> np.ndarray:
        """get contribution scores for input positions"""
        self.forward_fn.eval()
        
        combined_input = self._combine_inputs(seq_batch, dns_batch)
        
        if self.use_gpu:
            combined_input = combined_input.cuda()
        
        contribution_scores = self.forward_fn.get_contribution_scores(combined_input, target_class)
        
        return contribution_scores.cpu().numpy()
    
    def save_model(self, path: str):
        """save model state"""
        torch.save(self.forward_fn.state_dict(), path)
    
    def load_model(self, path: str):
        """load model state"""
        self.forward_fn.load_state_dict(torch.load(path, map_location='cpu'))
        if self.use_gpu:
            self.forward_fn = self.forward_fn.cuda()