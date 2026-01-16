"""Adapter to load a production S4/Mamba block if installed, else fall back to S4LiteBlock.

This module provides get_ssm_block(d_model, ...) which returns a nn.Module class
(or callable factory) that can be instantiated as the SSM block.
"""
from typing import Optional
import importlib
import torch.nn as nn


def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def get_ssm_block_factory(prefer: Optional[str] = None):
    """Return a factory function SSMFactory(d_model, **kwargs) -> nn.Module.

    prefer: Optional module name to prefer (e.g. 'mamba')
    """
    # Try user-preferred module first
    candidates = []
    if prefer:
        candidates.append(prefer)
    # common names for mamba/s4 packages
    candidates += ['mamba', 'mamba_ssm', 'state_spaces.mamba', 'state_spaces']

    for modname in candidates:
        mod = _try_import(modname)
        if mod is None:
            continue
        # Heuristics: look for plausible class names
        for attr in ['MambaBlock', 'S4Block', 'SSMBlock', 'StateSpaceBlock']:
            if hasattr(mod, attr):
                cls = getattr(mod, attr)

                def factory(d_model, *args, **kwargs):
                    return cls(d_model, *args, **kwargs)

                return factory
        # Some packages expose constructor via a function
        if hasattr(mod, 'create_ssm'):
            func = getattr(mod, 'create_ssm')

            def factory(d_model, *args, **kwargs):
                return func(d_model, *args, **kwargs)

            return factory

    # If none found, fall back to local S4LiteBlock defined in crsm.mamba_ssm
    try:
        # Use the vendor fallback implementation (no circular import)
        from .s4_vendor import S4LiteBlock

        def factory(d_model, *args, **kwargs):
            return S4LiteBlock(d_model, *args, **kwargs)

        return factory
    except Exception:
        # Last resort: simple linear passthrough
        class IdentitySSM(nn.Module):
            def __init__(self, d_model, *args, **kwargs):
                super().__init__()
                self.lin = nn.Identity()

            def forward(self, x, state=None):
                return x, None

        def factory(d_model, *args, **kwargs):
            return IdentitySSM(d_model)

        return factory
