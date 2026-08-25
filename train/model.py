"""nanoGPT-ish decoder. NO arm-specific logic here — see train/attention.py.

Trap §9.6 — embedding params dominate at this scale. At d_model=384 with
GPT-2's 50k vocab that is 19.3M embedding params against 10.6M in the actual
transformer, so the part under test becomes a minority of the model and the
softmax dominates compute. A 16k BPE trained on the project's own corpus drops
the total to ~17M and puts the layers under test back in the majority.
"""

from __future__ import annotations

import torch.nn as nn


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError

    def num_params(self, non_embedding: bool = True) -> int:
        """Report both. The non-embedding count is the one that means anything."""
        raise NotImplementedError
