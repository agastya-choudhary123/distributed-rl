"""Convert between PyTorch state dicts and wire-safe {str: np.ndarray} dicts.

Weights and gradients both travel over the wire as plain numpy dicts. Pickle
handles numpy arrays natively, so this module is only about the torch<->numpy
boundary. In production you'd swap pickle for MessagePack/flatbuffers here; the
numpy representation would stay the same.
"""

import numpy as np
import torch


def state_dict_to_numpy(state_dict: dict) -> dict:
    """{param_name: tensor} -> {param_name: np.ndarray}."""
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def numpy_to_state_dict(numpy_dict: dict) -> dict:
    """{param_name: np.ndarray} -> {param_name: tensor} for load_state_dict."""
    return {k: torch.from_numpy(np.array(v)) for k, v in numpy_dict.items()}
