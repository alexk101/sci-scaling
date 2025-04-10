import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

# Check if flash attention is available
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from .architecture import TransformerBlock, CustomNormalization


class PatchEmbed(nn.Module):
    """Patch embedding layer for Vision Transformer.

    Args:
        img_size: Tuple of (height, width) of input image
        patch_size: Size of patches to extract
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (
            (img_size[0] // patch_size) * (img_size[1] // patch_size)
        )

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Tensor of shape [B, N, E] where N is number of patches
        """
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional Flash Attention.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in qkv projection
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
        use_flash: Whether to use external Flash Attention
        use_native_flash: Whether to use PyTorch native Flash Attention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash: bool = True,
        use_native_flash: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        
        # Prioritize external flash attention when available
        self.use_flash = use_flash and FLASH_ATTN_AVAILABLE
        
        # Only use native flash attention if external package is not available
        self.use_native_flash = use_native_flash and not FLASH_ATTN_AVAILABLE

        # Check if PyTorch version supports scaled_dot_product_attention
        self.has_native_sdpa = hasattr(F, 'scaled_dot_product_attention')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, N, C]
            mask: Optional attention mask of shape [B, N, N]

        Returns:
            Tensor of shape [B, N, C]
        """
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Priority 1: Use external Flash Attention package if available and enabled
        if self.use_flash and mask is None:
            x = flash_attn_func(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            x = x.reshape(B, N, C)
        # Priority 2: Fall back to PyTorch native Flash Attention if external package not available
        elif self.use_native_flash and self.has_native_sdpa:
            # Reshape for PyTorch's scaled_dot_product_attention
            q_reshaped = q.transpose(1, 2)  # [B, nH, N, D]
            k_reshaped = k.transpose(1, 2)  # [B, nH, N, D]
            v_reshaped = v.transpose(1, 2)  # [B, nH, N, D]
            
            attn_mask = None
            if mask is not None:
                # Create attention mask compatible with scaled_dot_product_attention
                attn_mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            
            # Use PyTorch's scaled_dot_product_attention
            x = F.scaled_dot_product_attention(
                q_reshaped, 
                k_reshaped, 
                v_reshaped,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,  # Weather data typically doesn't need causal masking
            )
            
            x = x.transpose(1, 2).reshape(B, N, C)
        # Priority 3: Fall back to standard attention implementation
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation.

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (defaults to in_features)
        out_features: Output dimension (defaults to in_features)
        act_layer: Activation layer (defaults to GELU)
        drop: Dropout rate
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, N, C]

        Returns:
            Tensor of shape [B, N, C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WeatherTransformer(nn.Module):
    """Vision Transformer for weather prediction.

    Args:
        img_size: Tuple of (height, width) of input image
        patch_size: Size of patches to extract
        in_channels: Number of input channels
        out_channels: Number of output channels
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        mlp_ratio: MLP hidden dimension ratio
        qkv_bias: Whether to use bias in qkv projection
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        attention_type: Type of attention mechanism
        activation_type: Type of activation function
        normalization_type: Type of normalization
        use_pre_norm: Whether to use pre-norm transformer
        use_parallel_attention: Whether to use parallel attention
        use_flash_attention: Whether to use Flash Attention
        use_native_flash_attention: Whether to use PyTorch native Flash Attention
        use_checkpointing: Whether to use gradient checkpointing
        mixed_precision: Mixed precision training type
        use_kv_cache: Whether to use KV cache
        use_quantization: Whether to use quantization
        quantization_bits: Number of quantization bits
        use_fused_attention: Whether to use fused attention
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        # Architecture experiment parameters
        attention_type: str = "standard",
        activation_type: str = "gelu",
        normalization_type: str = "layer",
        use_pre_norm: bool = True,
        use_parallel_attention: bool = False,
        # Additional parameters
        use_flash_attention: bool = True,
        use_native_flash_attention: bool = False,
        use_checkpointing: bool = False,
        mixed_precision: str = "bf16",
        use_kv_cache: bool = False,
        use_quantization: bool = False,
        quantization_bits: int = 8,
        use_fused_attention: bool = True,
    ):
        super().__init__()
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.use_kv_cache = use_kv_cache
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        
        # Flash attention configuration - prioritize external package when available
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE
        self.use_native_flash_attention = use_native_flash_attention and not FLASH_ATTN_AVAILABLE
        
        if use_flash_attention and not FLASH_ATTN_AVAILABLE and use_native_flash_attention:
            logging.info("External Flash Attention package not available, falling back to PyTorch native implementation")
        elif use_flash_attention and FLASH_ATTN_AVAILABLE:
            logging.info("Using external Flash Attention package")
        elif use_native_flash_attention and not FLASH_ATTN_AVAILABLE:
            logging.warning("Native Flash Attention requested but FLASH_ATTN_AVAILABLE is False - may not be available")

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, num_layers)
        ]

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    attention_type=attention_type,
                    activation_type=activation_type,
                    normalization_type=normalization_type,
                    use_pre_norm=use_pre_norm,
                    use_parallel_attention=use_parallel_attention,
                    window_size=num_patches,
                    num_groups=8,
                )
                for i in range(num_layers)
            ]
        )

        # Output head
        self.out_size = out_channels * patch_size * patch_size
        self.norm = CustomNormalization(embed_dim, normalization_type)
        self.head = nn.Linear(embed_dim, self.out_size, bias=False)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        # Enable gradient checkpointing if requested
        if use_checkpointing:
            self.gradient_checkpointing_enable()

        # Enable flash attention if requested
        if use_flash_attention:
            self._enable_flash_attention()

        # Store grid dimensions for reshaping during inference
        self.patch_h = img_size[0] // patch_size
        self.patch_w = img_size[1] // patch_size
        self.out_channels = out_channels

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _enable_flash_attention(self):
        """Enable flash attention for all transformer blocks."""
        for block in self.blocks:
            if hasattr(block.attn, "use_flash"):
                block.attn.use_flash = self.use_flash_attention
            
            # For our modified MultiHeadAttention class
            if hasattr(block.attn, "use_native_flash"):
                block.attn.use_native_flash = self.use_native_flash_attention
                
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        logging.info("Enabling gradient checkpointing for memory efficiency")
        for block in self.blocks:
            block.enable_checkpointing()

    def forward_head(self, x: torch.Tensor, original_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """Forward head using the reshape approach.
        
        Args:
            x: Input tensor of shape [B, N, embed_dim]
            original_shape: Original input shape (B, C, H, W)
            
        Returns:
            Tensor of shape [B, out_channels, H, W]
        """
        B, _, _ = x.shape  # B x N x embed_dim

        # Reshape to [B, h, w, embed_dim]
        x = x.reshape(B, self.patch_h, self.patch_w, self.embed_dim)
        
        # Apply head (linear projection)
        x = self.head(x)  # [B, h, w, out_channels*patch_size*patch_size]
        
        # Reshape to unfold the patches
        patch_size = self.patch_embed.patch_size
        x = x.reshape(B, self.patch_h, self.patch_w, patch_size, patch_size, self.out_channels)
        
        # Rearrange dimensions using einsum for efficiency
        x = torch.einsum("nhwpqc->nchpwq", x)
        
        # Final reshape to output dimensions [B, C, H, W]
        output_H = self.patch_h * patch_size
        output_W = self.patch_w * patch_size
        x = x.reshape(B, self.out_channels, output_H, output_W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Tensor of shape [B, out_channels, H, W]
        """
        # Store original spatial dimensions
        original_shape = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output head
        x = self.norm(x)
        
        # Use the new reshaping approach
        x = self.forward_head(x, original_shape)

        # Quantize output if requested
        if self.use_quantization:
            x = self._quantize(x)

        return x

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize the output tensor."""
        if self.quantization_bits == 8:
            return torch.quantize_per_tensor(
                x, scale=1.0, zero_point=0, dtype=torch.qint8
            )
        elif self.quantization_bits == 16:
            return torch.quantize_per_tensor(
                x, scale=1.0, zero_point=0, dtype=torch.qint16
            )
        elif self.quantization_bits == 4:
            return torch.quantize_per_tensor(
                x, scale=1.0, zero_point=0, dtype=torch.qint4
            )
        else:
            raise ValueError(
                f"Unsupported quantization bits: {self.quantization_bits}"
            )

    def get_attention_maps(self) -> list:
        """Get attention maps from all transformer blocks."""
        attention_maps = []
        for block in self.blocks:
            if hasattr(block.attn, "attn"):
                attention_maps.append(block.attn.attn)
        return attention_maps
