"""
TCN (Temporal Convolutional Network) model for RUL prediction.

Implements the frozen architecture from configs/tcn_spec.yaml:
- 3 TCN blocks with 64 filters, kernel=3, dilations=[1,2,4]
- BatchNorm1D + ReLU + Dropout(0.2)
- GlobalAveragePooling1D
- Head: Dense(64, ReLU) → Dense(1)
"""
import torch
import torch.nn as nn
from typing import List, Optional


class TCNBlock(nn.Module):
    """
    Single TCN block with dilated causal convolution.
    
    Structure: Conv1D → BatchNorm1D → ReLU → Dropout → Conv1D → BatchNorm1D → ReLU → Dropout
    With residual connection if input/output channels differ.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Causal padding: (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        
        # First convolution
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolution
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if channels differ)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu_out = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        # Store for residual
        residual = self.residual(x)
        
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Causal trim
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Causal trim
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        return self.relu_out(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for RUL prediction.
    
    Architecture per tcn_spec.yaml:
    - 3 blocks, 64 filters/block, kernel=3, dilations=[1,2,4]
    - GlobalAveragePooling1D
    - Dense(64, ReLU) → Dense(1)
    """
    
    def __init__(
        self,
        input_channels: int,
        num_blocks: int = 3,
        filters: int = 64,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.2,
        head_units: int = 64
    ):
        super().__init__()
        
        if dilations is None:
            dilations = [1, 2, 4]
        
        assert len(dilations) == num_blocks, "Number of dilations must match num_blocks"
        
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        self.filters = filters
        
        # TCN blocks
        blocks = []
        in_ch = input_channels
        for i, dilation in enumerate(dilations):
            blocks.append(TCNBlock(
                in_channels=in_ch,
                out_channels=filters,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
            in_ch = filters
        self.tcn_blocks = nn.Sequential(*blocks)
        
        # Global average pooling (over time dimension)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Head: Dense(64, ReLU) → Dense(1)
        self.head = nn.Sequential(
            nn.Linear(filters, head_units),
            nn.ReLU(),
            nn.Linear(head_units, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features) - time series input
        Returns:
            (batch,) - RUL predictions
        """
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # TCN blocks
        x = self.tcn_blocks(x)
        
        # Global average pooling: (batch, filters, seq_len) → (batch, filters, 1)
        x = self.global_pool(x)
        
        # Flatten and head
        x = x.squeeze(-1)  # (batch, filters)
        x = self.head(x)   # (batch, 1)
        
        return x.squeeze(-1)  # (batch,)


def create_tcn_model(input_channels: int = 17) -> TCN:
    """
    Create TCN model with frozen spec parameters.
    
    Args:
        input_channels: Number of input features (default 17 = 14 sensors + 3 ops)
    
    Returns:
        TCN model instance
    """
    return TCN(
        input_channels=input_channels,
        num_blocks=3,
        filters=64,
        kernel_size=3,
        dilations=[1, 2, 4],
        dropout=0.2,
        head_units=64
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
