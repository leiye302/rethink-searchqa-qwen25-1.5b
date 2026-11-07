"""Native FP32 AdamW with all training state on the model's CUDA device.

This implements the same global-batch interface as CpuAdamWOffload, without
gradient hooks, host masters, intermediate updates, or host gradient transfers.
"""
from __future__ import annotations
import torch


class GpuResidentAdamW:
    def __init__(self, model, *, lr=2e-6, betas=(.9, .95), weight_decay=.01, foreach=True):
        self.named = list(model.named_parameters())
        if not self.named or not all(p.requires_grad and p.dtype == torch.float32 for _, p in self.named):
            raise ValueError('Full trainable FP32 parameters are required')
        devices = {p.device for _, p in self.named}
        if len(devices) != 1:
            raise ValueError('Every parameter must share one device')
        self.optimizer = torch.optim.AdamW([p for _, p in self.named], lr=lr, betas=betas,
                                          weight_decay=weight_decay, eps=1e-8, foreach=foreach)
        self.active = False
        self.updates = 0

    def begin(self):
        if self.active:
            raise RuntimeError('Global batch is already open')
        self.optimizer.zero_grad(set_to_none=True)
        self.active = True

    def finish(self, clip=1.):
        parameters = [p for _, p in self.named]
        if not self.active or any(p.grad is None for p in parameters):
            raise RuntimeError('Incomplete full-parameter gradient batch')
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True, foreach=True)
        self.optimizer.step()
        self.active = False
        self.updates += 1
        return float(norm)

    def release_gradients(self):
        if self.active:
            raise RuntimeError('Cannot release an active batch')
        self.optimizer.zero_grad(set_to_none=True)

    def close(self):
        # No hooks, additional masters, services, or external resources exist.
        pass
