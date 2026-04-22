from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch


@dataclass
class EntityFeature:
    name: str
    hidden_list: List[torch.Tensor]
    act_list: List[torch.Tensor]
    h_bar: torch.Tensor
    k_bar: torch.Tensor
    h_var: float
    k_var: float
    strength: float
    meta: Optional[Dict[str, Any]] = None


@dataclass
class EntityFeatureSummary:
    name: str
    h_var: float
    k_var: float
    strength: float