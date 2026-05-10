#!/usr/bin/env python3
import os
from typing import Optional

import torch


def dir_size_bytes(path: Optional[str]) -> Optional[int]:
    if not path or not os.path.isdir(path):
        return None
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                total += os.path.getsize(fpath)
            except OSError:
                continue
    return int(total)


def estimate_fp16_weight_bytes(model: torch.nn.Module, bytes_per_param: int = 2) -> int:
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return int(num_params) * int(bytes_per_param)
