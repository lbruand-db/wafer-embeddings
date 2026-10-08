"""DINO self-distillation head, loss, and EMA teacher (SPECS.md §4, ref [1]).

The served embedding is the **encoder** output (384-d, §3 N1); the DINO head
(prototype logits) is used only during pretraining and discarded afterwards.
"""

from __future__ import annotations

import copy
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """DINO projection head: MLP -> L2-normalized bottleneck -> cosine-to-prototypes.

    The prototype (last-layer) weights are **unit-normalized** in the forward pass,
    so logits are cosine similarities in [-1, 1] (equivalent to DINO's weight-norm
    with the magnitude frozen at 1). A plain Linear here leaves logits ~0 and the loss
    pinned at ln(out_dim) — the uniform collapse we observed (ref [1]).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden: int = 2048,
        bottleneck: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(max(n_layers - 2, 0)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, bottleneck))
        self.mlp = nn.Sequential(*layers)
        self.prototypes = nn.Linear(bottleneck, out_dim, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.mlp(x), dim=-1)
        w = F.normalize(self.prototypes.weight, dim=1)  # unit-norm prototypes
        return F.linear(x, w)


class DINOLoss(nn.Module):
    """Teacher-student cross-entropy with teacher centering (ref [1])."""

    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self, student_outputs: list[torch.Tensor], teacher_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        center = cast(torch.Tensor, self.center)
        student = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_outputs]
        teacher = [
            F.softmax((t - center) / self.teacher_temp, dim=-1).detach() for t in teacher_outputs
        ]
        total = student_outputs[0].new_zeros(())
        n_terms = 0
        for ti, t in enumerate(teacher):
            for si, s in enumerate(student):
                if si == ti:
                    continue  # a view is never distilled against itself
                total = total + (-(t * s).sum(dim=-1).mean())
                n_terms += 1
        self._update_center(torch.cat(teacher_outputs, dim=0))
        return total / max(n_terms, 1)

    @torch.no_grad()
    def _update_center(self, teacher_cat: torch.Tensor) -> None:
        batch_center = teacher_cat.mean(dim=0, keepdim=True)
        center = cast(torch.Tensor, self.center)
        center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


class DinoModel(nn.Module):
    """Student (trained) + EMA teacher, each = encoder + DINO head.

    ``embed`` returns the student **encoder** embedding — the artifact we serve.
    """

    def __init__(self, encoder: nn.Module, head: DINOHead, teacher_momentum: float = 0.996):
        super().__init__()
        self.teacher_momentum = teacher_momentum
        self.student_enc = encoder
        self.student_head = head
        self.teacher_enc = copy.deepcopy(encoder)
        self.teacher_head = copy.deepcopy(head)
        for p in list(self.teacher_enc.parameters()) + list(self.teacher_head.parameters()):
            p.requires_grad_(False)

    def embed(self, coords, state_ids, mask) -> torch.Tensor:
        return self.student_enc(coords, state_ids, mask)

    def student_views(self, views: list[tuple]) -> list[torch.Tensor]:
        return [self.student_head(self.student_enc(c, s, m)) for (c, s, m) in views]

    @torch.no_grad()
    def teacher_views(self, views: list[tuple]) -> list[torch.Tensor]:
        return [self.teacher_head(self.teacher_enc(c, s, m)) for (c, s, m) in views]

    @torch.no_grad()
    def update_teacher(self) -> None:
        m = self.teacher_momentum
        pairs = [
            (self.student_enc, self.teacher_enc),
            (self.student_head, self.teacher_head),
        ]
        for student, teacher in pairs:
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1 - m)
