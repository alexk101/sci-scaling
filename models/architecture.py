import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # Add this for gradient checkpointing


class CustomAttention(nn.Module):
    """Base class for attention mechanisms with different implementations."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attention_type: str = "standard",
        window_size: int = 32,
        sparsity_ratio: float = 0.5,
        linear_heads: int = 4,
        use_flash: bool = False,  # Use external Flash Attention package
        use_native_flash: bool = False,  # Use PyTorch native Flash Attention
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attention_type = attention_type
        
        # Check if external flash attention is available
        try:
            from flash_attn import flash_attn_func
            FLASH_ATTN_AVAILABLE = True
        except ImportError:
            FLASH_ATTN_AVAILABLE = False
            
        # Prioritize external flash attention when available
        self.use_flash = use_flash and FLASH_ATTN_AVAILABLE
        
        # Only use native flash attention if external package is not available
        self.use_native_flash = use_native_flash and not FLASH_ATTN_AVAILABLE
        
        # Check if PyTorch version supports scaled_dot_product_attention
        self.has_native_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # Common components
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Architecture-specific parameters
        self.window_size = window_size
        self.sparsity_ratio = sparsity_ratio
        self.linear_heads = linear_heads

        # Initialize architecture-specific components
        self._init_architecture()

    def _init_architecture(self):
        """Initialize architecture-specific components."""
        if self.attention_type == "linear":
            self.linear_proj = nn.Linear(self.num_heads, self.linear_heads)
        elif self.attention_type == "sparse":
            self.sparse_mask = self._create_sparse_mask()
        elif self.attention_type == "sliding_window":
            self.window_mask = self._create_window_mask()
        elif self.attention_type == "global_local":
            self.global_mask = self._create_global_local_mask()

    def _create_sparse_mask(self) -> torch.Tensor:
        """Create sparse attention mask."""
        seq_len = self.window_size
        mask = torch.ones(seq_len, seq_len)
        num_zeros = int(seq_len * seq_len * self.sparsity_ratio)
        zero_indices = torch.randperm(seq_len * seq_len)[:num_zeros]
        mask.view(-1)[zero_indices] = 0
        return mask

    def _create_window_mask(self) -> torch.Tensor:
        """Create sliding window attention mask."""
        seq_len = self.window_size
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask

    def _create_global_local_mask(self) -> torch.Tensor:
        """Create global-local attention mask."""
        seq_len = self.window_size
        mask = torch.ones(seq_len, seq_len)
        # Global attention for first and last tokens
        mask[0, :] = 1
        mask[-1, :] = 1
        # Local attention for middle tokens
        for i in range(1, seq_len - 1):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # Try to use external Flash Attention if available and the attention type is standard
        if self.use_flash and self.attention_type == "standard":
            try:
                from flash_attn import flash_attn_func
                x = flash_attn_func(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.0
                )
                x = x.transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x
            except (ImportError, Exception) as e:
                # If there's an error using flash_attn, fall through to the next method
                pass
        
        # Try to use native PyTorch Flash Attention if enabled and available and external not used
        if self.use_native_flash and self.has_native_sdpa and self.attention_type == "standard":
            # PyTorch's scaled_dot_product_attention expects inputs in shape [B, num_heads, seq_len, head_dim]
            # which is the shape we already have
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False
            )
            x = attn_output.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
        # Fall back to standard attention or specialized attention types
        if self.attention_type == "standard":
            attn = (q @ k.transpose(-2, -1)) * self.scale
        elif self.attention_type == "linear":
            # Linear attention: O(n) complexity
            q = self.linear_proj(q.transpose(1, 2)).transpose(1, 2)
            k = self.linear_proj(k.transpose(1, 2)).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * self.scale
        elif self.attention_type == "sparse":
            # Sparse attention with mask
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn * self.sparse_mask.to(attn.device)
        elif self.attention_type == "sliding_window":
            # Sliding window attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn * self.window_mask.to(attn.device)
        elif self.attention_type == "global_local":
            # Global-local attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn * self.global_mask.to(attn.device)
        else:
            raise ValueError(
                f"Unknown attention type: {self.attention_type}"
            )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CustomActivation(nn.Module):
    """Custom activation functions for the transformer."""

    def __init__(self, activation_type: str = "gelu"):
        super().__init__()
        self.activation_type = activation_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "gelu":
            return F.gelu(x)
        elif self.activation_type == "relu":
            return F.relu(x)
        elif self.activation_type == "selu":
            return F.selu(x)
        elif self.activation_type == "swish":
            return x * torch.sigmoid(x)
        elif self.activation_type == "mish":
            return x * torch.tanh(F.softplus(x))
        else:
            raise ValueError(
                f"Unknown activation type: {self.activation_type}"
            )


class CustomNormalization(nn.Module):
    """Custom normalization layers for the transformer."""

    def __init__(
        self,
        dim: int,
        normalization_type: str = "layer",
        eps: float = 1e-5,
        num_groups: int = 8,
    ):
        super().__init__()
        self.normalization_type = normalization_type
        self.eps = eps

        if normalization_type == "layer":
            self.norm = nn.LayerNorm(dim, eps=eps)
        elif normalization_type == "rms":
            self.norm = RMSNorm(dim, eps=eps)
        elif normalization_type == "group":
            self.norm = nn.GroupNorm(num_groups, dim, eps=eps)
        elif normalization_type == "instance":
            self.norm = nn.InstanceNorm1d(dim, eps=eps)
        else:
            raise ValueError(
                f"Unknown normalization type: {normalization_type}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization_type in ["layer", "rms"]:
            return self.norm(x)
        elif self.normalization_type in ["group", "instance"]:
            # Reshape for 1D normalization
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, N)
            x = self.norm(x)
            return x.reshape(B, C, N).transpose(1, 2)
        else:
            raise ValueError(
                f"Unknown normalization type: {self.normalization_type}"
            )


class RMSNorm(nn.Module):
    """Root Mean Square normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class TransformerBlock(nn.Module):
    """Transformer block with configurable architecture components."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        attention_type: str = "standard",
        activation_type: str = "gelu",
        normalization_type: str = "layer",
        use_pre_norm: bool = True,
        use_parallel_attention: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.use_pre_norm = use_pre_norm
        self.use_parallel_attention = use_parallel_attention

        # Filter normalization parameters
        norm_kwargs = {
            'eps': kwargs.get('eps', 1e-5),
            'num_groups': kwargs.get('num_groups', 8)
        }
        
        # Attention parameters (window_size, sparsity_ratio, linear_heads)
        attn_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['window_size', 'sparsity_ratio', 'linear_heads']}
        
        # Normalization layers
        self.norm1 = CustomNormalization(
            dim, normalization_type, **norm_kwargs
        )
        self.norm2 = CustomNormalization(
            dim, normalization_type, **norm_kwargs
        )

        # Attention
        self.attn = CustomAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            attention_type=attention_type,
            **attn_kwargs,
        )

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            CustomActivation(activation_type),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        # Drop path
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pre_norm:
            if self.use_parallel_attention:
                # Parallel attention and MLP
                x = (
                    x
                    + self.drop_path(self.attn(self.norm1(x)))
                    + self.drop_path(self.mlp(self.norm2(x)))
                )
            else:
                # Sequential attention and MLP
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.use_parallel_attention:
                # Parallel attention and MLP
                x = self.norm1(x + self.drop_path(self.attn(x)))
                x = self.norm2(x + self.drop_path(self.mlp(x)))
            else:
                # Sequential attention and MLP
                x = self.norm1(x + self.drop_path(self.attn(x)))
                x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x
        
    def enable_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        # Store the original forward method
        self._original_forward = self.forward
        
        # Define a checkpointed forward method
        def checkpointed_forward(x):
            # Ensure x requires gradients for checkpointing to work
            needs_grad = x.requires_grad
            if not needs_grad:
                x.requires_grad = True
            output = torch.utils.checkpoint.checkpoint(self._original_forward, x, use_reentrant=False)
            if not needs_grad:
                output = output.detach()
            return output
        
        # Replace the forward method with the checkpointed version
        self.forward = checkpointed_forward


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (
            keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        )
        random_tensor.floor_()  # binarize
        output = (
            x.div(keep_prob) * random_tensor
            if self.scale_by_keep
            else x * random_tensor
        )
        return output
