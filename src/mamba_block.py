# src/mamba_block.py
"""
MambaBlock implementation extracted from notebooks/04_implement_mamba.ipynb.

This module:
- imports the Mamba operator from the `mamba_ssm` package,
- defines the exact MambaBlock used in the notebook (class name: _MambaBlockImpl),
- exposes a flexible wrapper class MambaBlock that accepts both:
    * the notebook signature: MambaBlock(in_c, d_model, d_state=16, d_conv=4, expand=2)
    * the older placeholder-style signature: MambaBlock(channels, kernel=31)  -> mapped to d_model=channels

If `mamba_ssm` is not installed, importing this module will raise a helpful ImportError.
"""

from typing import Any
import torch
import torch.nn as nn

# Try to import the real Mamba operator
try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError(
        "Failed to import `mamba_ssm.Mamba`. Please install the package used in the notebook, e.g.:\n\n"
        "    pip install --no-build-isolation mamba-ssm==2.2.4\n\n"
        f"Original error: {e}"
    ) from e


class _MambaBlockImpl(nn.Module):
    """
    Exact MambaBlock implementation (kept as the notebook-defined class).
    Signature:
        _MambaBlockImpl(in_c, d_model, d_state=16, d_conv=4, expand=2)
    - in_c: input channels (int)
    - d_model: model dimension used by Mamba (int)
    - d_state, d_conv, expand: hyperparameters forwarded to Mamba(...)
    """
    def __init__(self, in_c: int, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.in_c = in_c
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # project channels -> model dim
        self.proj = nn.Linear(in_c, d_model)
        # the actual Mamba operator (state-space style)
        self.mamba = Mamba(d_model, d_state, d_conv, expand)
        # normalization and projection back to channels
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, in_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Steps:
         - permute and flatten spatial dims -> (B, H*W, C)
         - project to d_model, layernorm, apply Mamba, project back
         - reshape back to (B, C, H, W)
         - residual add with input
        """
        b, c, h, w = x.shape
        # flatten spatial dims: (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # project and normalize
        x_mamba = self.proj(x_flat)          # (B, L, d_model)
        x_mamba = self.norm(x_mamba)
        # apply Mamba operator
        x_mamba = self.mamba(x_mamba)
        # project back to channels
        x_mamba = self.proj_out(x_mamba)     # (B, L, C)

        # restore spatial layout
        x_out = x_mamba.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x_out + x  # residual


class MambaBlock(nn.Module):
    """
    Backwards-compatible wrapper that instantiates the notebook Mamba block.

    Accepts either:
      1) Notebook-style args:
           MambaBlock(in_c, d_model, d_state=16, d_conv=4, expand=2)
         (recommended — matches the notebook exactly)

      2) Placeholder-style args (keeps compatibility with earlier placeholder usage):
           MambaBlock(channels, kernel=31)
         In this case we map:
           d_model = channels
           d_state = 16
           d_conv  = 4
           expand  = 2
         (the `kernel` argument is ignored but accepted for compatibility)

    This wrapper ensures older calls like MambaBlock(channels, kernel=31) won't crash.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

        # If user passed keyword arguments for notebook style, forward them
        if "d_model" in kwargs or (len(args) >= 2):
            # Notebook-style: expect (in_c, d_model, ...)
            in_c = args[0] if len(args) >= 1 else kwargs.get("in_c")
            d_model = args[1] if len(args) >= 2 else kwargs.get("d_model")
            d_state = kwargs.get("d_state", 16) if len(args) < 3 else (args[2] if len(args) >= 3 else 16)
            d_conv = kwargs.get("d_conv", 4) if len(args) < 4 else (args[3] if len(args) >= 4 else 4)
            expand = kwargs.get("expand", 2) if len(args) < 5 else (args[4] if len(args) >= 5 else 2)
            if in_c is None or d_model is None:
                raise ValueError("Invalid arguments. For notebook-style usage pass in_c and d_model.")
            self.impl = _MambaBlockImpl(int(in_c), int(d_model), int(d_state), int(d_conv), int(expand))
        else:
            # Placeholder-style: MambaBlock(channels, kernel=31) or MambaBlock(channels)
            # Map this to notebook impl with d_model = channels (reasonable default)
            channels = args[0] if len(args) >= 1 else kwargs.get("channels")
            if channels is None:
                raise ValueError("Missing required 'channels' argument for MambaBlock.")
            # use defaults for other hyperparams
            d_model = int(channels)
            d_state = int(kwargs.get("d_state", 16))
            d_conv = int(kwargs.get("d_conv", 4))
            expand = int(kwargs.get("expand", 2))
            self.impl = _MambaBlockImpl(int(channels), d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)


# Convenience export
__all__ = ["MambaBlock", "_MambaBlockImpl"]
