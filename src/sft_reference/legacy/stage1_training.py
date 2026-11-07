"""Task-balanced sampling and loss reduction for Rethink Stage-1 SFT."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Sampler


SFT_LOSS_REDUCTIONS = ("token_mean", "sample_mean")


def reasoning_reference_kl_loss(
    student_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    reasoning_mask: torch.Tensor,
) -> torch.Tensor:
    """Return sample-mean forward KL on masked reasoning predictions.

    The frozen reference distribution is the target.  The mask follows the
    trainer convention: ``reasoning_mask[:, j]`` selects the prediction of the
    token at ``input_ids[:, j + 1]``.  Suffix SFT and reasoning preservation
    therefore stay disjoint while sharing one causal forward sequence.
    """

    if student_logits.ndim != 3 or reference_logits.ndim != 3:
        raise ValueError("student_logits and reference_logits must be [batch, sequence, vocab].")
    if student_logits.shape != reference_logits.shape:
        raise ValueError(
            f"student shape {tuple(student_logits.shape)} does not match "
            f"reference shape {tuple(reference_logits.shape)}."
        )
    if reasoning_mask.shape != student_logits.shape[:2]:
        raise ValueError(
            f"reasoning_mask shape {tuple(reasoning_mask.shape)} does not match "
            f"logit prefix {tuple(student_logits.shape[:2])}."
        )

    per_sample = []
    for student_row, reference_row, mask_row in zip(
        student_logits, reference_logits, reasoning_mask, strict=True
    ):
        selected = mask_row.to(device=student_row.device, dtype=torch.bool)
        if not torch.any(selected):
            continue
        student_log_probs = torch.log_softmax(student_row[selected].float(), dim=-1)
        reference_log_probs = torch.log_softmax(reference_row[selected].float(), dim=-1)
        reference_probs = reference_log_probs.exp()
        token_kl = torch.sum(
            reference_probs * (reference_log_probs - student_log_probs), dim=-1
        )
        per_sample.append(token_kl.mean())

    if not per_sample:
        return student_logits.sum() * 0.0
    return torch.stack(per_sample).mean()


def masked_sft_loss(
    token_losses: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    reduction: str,
) -> torch.Tensor:
    """Reduce ``[batch, sequence]`` token losses.

    ``sample_mean`` implements response-token mean within each sample followed
    by a mean across non-empty samples.  Longer Analyzer JSON targets therefore
    do not outweigh shorter Actor action targets merely because of length.
    """

    if token_losses.ndim != 2 or loss_mask.ndim != 2:
        raise ValueError("token_losses and loss_mask must both have shape [batch, sequence].")
    if token_losses.shape != loss_mask.shape:
        raise ValueError(
            f"token_losses shape {tuple(token_losses.shape)} does not match "
            f"loss_mask shape {tuple(loss_mask.shape)}."
        )
    reduction = str(reduction)
    if reduction not in SFT_LOSS_REDUCTIONS:
        raise ValueError(
            f"Unsupported SFT loss reduction {reduction!r}; expected {SFT_LOSS_REDUCTIONS}."
        )

    mask = loss_mask.to(device=token_losses.device, dtype=token_losses.dtype)
    masked_losses = token_losses * mask
    if reduction == "token_mean":
        return masked_losses.sum() / mask.sum().clamp_min(1.0)

    token_counts = mask.sum(dim=-1)
    valid_samples = (token_counts > 0).to(dtype=token_losses.dtype)
    per_sample_loss = masked_losses.sum(dim=-1) / token_counts.clamp_min(1.0)
    return (per_sample_loss * valid_samples).sum() / valid_samples.sum().clamp_min(1.0)


class TaskBalancedDistributedBatchSampler(Sampler[List[int]]):
    """Yield globally task-balanced and optionally route-stratified batches.

    The union of the local batches yielded at a step contains the same number
    of examples from each configured task kind.  A rank may receive only one
    row (for example global batch 8 over 8 ranks), so balance is enforced at
    the global batch boundary.  When ``routes``/``route_values`` are supplied,
    the configured ``route_task_kind`` share is further divided equally among
    routes.  Thus the Stage-1 two-task/two-route setup with global batch eight
    yields four Analyzer, two Actor-Good, and two Actor-Bad rows.  Smaller pools
    are deterministically cycled.
    """

    def __init__(
        self,
        *,
        task_kinds: Sequence[str],
        task_kind_values: Sequence[str],
        routes: Optional[Sequence[str]] = None,
        route_values: Optional[Sequence[str]] = None,
        route_task_kind: Optional[str] = None,
        route_batch_counts: Optional[Mapping[str, int]] = None,
        epoch_anchor_route: Optional[str] = None,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        self.task_kinds = [str(kind) for kind in task_kinds]
        self.task_kind_values = tuple(str(kind) for kind in task_kind_values)
        self.route_values = tuple(str(route) for route in (route_values or ()))
        self.route_task_kind = (
            str(route_task_kind or self.task_kind_values[-1])
            if self.route_values and self.task_kind_values
            else None
        )
        self.route_batch_counts = {
            str(route): int(count)
            for route, count in dict(route_batch_counts or {}).items()
        }
        self.epoch_anchor_route = (
            str(epoch_anchor_route) if epoch_anchor_route is not None else None
        )
        self.routes = [str(route) for route in routes] if routes is not None else None
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        if not self.task_kind_values:
            raise ValueError("task_kind_values must contain at least one task kind.")
        if len(set(self.task_kind_values)) != len(self.task_kind_values):
            raise ValueError("task_kind_values must be unique.")
        if len(set(self.route_values)) != len(self.route_values):
            raise ValueError("route_values must be unique.")
        if self.route_values and self.routes is None:
            raise ValueError("routes must be supplied when route_values are configured.")
        if self.routes is not None and not self.route_values:
            raise ValueError("route_values must be supplied when routes are configured.")
        if self.route_batch_counts and not self.route_values:
            raise ValueError("route_batch_counts requires configured route_values.")
        if self.epoch_anchor_route is not None and not self.route_values:
            raise ValueError("epoch_anchor_route requires configured route_values.")
        if self.routes is not None and len(self.routes) != len(self.task_kinds):
            raise ValueError("routes and task_kinds must have the same length.")
        if self.route_values and self.route_task_kind not in self.task_kind_values:
            raise ValueError(
                f"route_task_kind={self.route_task_kind!r} must be in task_kind_values."
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must satisfy 0 <= rank < num_replicas.")

        self.global_batch_size = self.batch_size * self.num_replicas
        if self.global_batch_size % len(self.task_kind_values) != 0:
            raise ValueError(
                f"Global batch_size={self.global_batch_size} must be divisible by "
                f"{len(self.task_kind_values)} task kinds."
            )

        configured_kinds = frozenset(self.task_kind_values)
        observed_kinds = frozenset(self.task_kinds)
        unexpected = observed_kinds - configured_kinds
        if unexpected:
            raise ValueError(
                "Dataset contains task kinds outside task_kind_values: "
                + ", ".join(sorted(unexpected))
            )

        self.examples_per_kind_global = self.global_batch_size // len(self.task_kind_values)
        if self.route_batch_counts:
            if set(self.route_batch_counts) != set(self.route_values):
                raise ValueError(
                    "route_batch_counts keys must exactly match route_values."
                )
            if any(count <= 0 for count in self.route_batch_counts.values()):
                raise ValueError("Every route_batch_counts quota must be positive.")
            if sum(self.route_batch_counts.values()) != self.examples_per_kind_global:
                raise ValueError(
                    "route_batch_counts must sum to the global share of "
                    f"route_task_kind={self.route_task_kind!r}: "
                    f"{self.examples_per_kind_global}."
                )
        elif (
            self.route_values
            and self.examples_per_kind_global % len(self.route_values) != 0
        ):
            raise ValueError(
                f"The global share for route_task_kind={self.route_task_kind!r} is "
                f"{self.examples_per_kind_global}, which must be divisible by "
                f"{len(self.route_values)} route values. For two task kinds and two "
                "routes, global batch_size must be divisible by 4."
            )

        self.indices_by_stratum: Dict[Tuple[str, Optional[str]], List[int]] = defaultdict(list)
        unexpected_routes = set()
        for index, kind in enumerate(self.task_kinds):
            if self.route_values and kind == self.route_task_kind:
                route = self.routes[index]  # type: ignore[index]
                if route not in self.route_values:
                    unexpected_routes.add(route)
                    continue
                stratum = (kind, route)
            else:
                stratum = (kind, None)
            self.indices_by_stratum[stratum].append(index)
        if unexpected_routes:
            raise ValueError(
                f"Rows for route_task_kind={self.route_task_kind!r} contain routes "
                "outside route_values: " + ", ".join(sorted(unexpected_routes))
            )

        self.strata: List[Tuple[str, Optional[str]]] = []
        self.quota_by_stratum: Dict[Tuple[str, Optional[str]], int] = {}
        for kind in self.task_kind_values:
            if self.route_values and kind == self.route_task_kind:
                for route in self.route_values:
                    stratum = (kind, route)
                    self.strata.append(stratum)
                    self.quota_by_stratum[stratum] = (
                        self.route_batch_counts[route]
                        if self.route_batch_counts
                        else self.examples_per_kind_global // len(self.route_values)
                    )
            else:
                stratum = (kind, None)
                self.strata.append(stratum)
                self.quota_by_stratum[stratum] = self.examples_per_kind_global

        missing = [stratum for stratum in self.strata if not self.indices_by_stratum[stratum]]
        if missing:
            raise ValueError(
                "Balanced sampling requires at least one row for every sampling stratum; missing "
                + ", ".join(self._stratum_name(stratum) for stratum in missing)
            )

        if self.epoch_anchor_route is not None:
            if self.epoch_anchor_route not in self.route_values:
                raise ValueError("epoch_anchor_route must be one of route_values.")
            anchor = (self.route_task_kind, self.epoch_anchor_route)
            self.num_batches = max(
                1,
                math.ceil(
                    len(self.indices_by_stratum[anchor])
                    / self.quota_by_stratum[anchor]
                ),
            )
        else:
            self.num_batches = max(
                1,
                max(
                    math.ceil(
                        len(self.indices_by_stratum[stratum])
                        / self.quota_by_stratum[stratum]
                    )
                    for stratum in self.strata
                ),
            )

    @staticmethod
    def _stratum_name(stratum: Tuple[str, Optional[str]]) -> str:
        kind, route = stratum
        return kind if route is None else f"{kind}/{route}"

    @property
    def source_counts(self) -> Dict[str, int]:
        return dict(Counter(self.task_kinds))

    @property
    def stratum_source_counts(self) -> Dict[str, int]:
        return {
            self._stratum_name(stratum): len(self.indices_by_stratum[stratum])
            for stratum in self.strata
        }

    @property
    def stratum_batch_counts(self) -> Dict[str, int]:
        return {
            self._stratum_name(stratum): self.quota_by_stratum[stratum]
            for stratum in self.strata
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _cycle_shuffled(
        indices: Sequence[int],
        *,
        target_length: int,
        rng: random.Random,
    ) -> List[int]:
        expanded: List[int] = []
        while len(expanded) < target_length:
            cycle = list(indices)
            rng.shuffle(cycle)
            expanded.extend(cycle)
        return expanded[:target_length]

    def __iter__(self) -> Iterator[List[int]]:
        expanded_by_stratum: Dict[Tuple[str, Optional[str]], List[int]] = {}
        for stratum_index, stratum in enumerate(self.strata):
            quota = self.quota_by_stratum[stratum]
            rng = random.Random(
                self.seed + self.epoch * 1_000_003 + stratum_index * 10_007
            )
            expanded_by_stratum[stratum] = self._cycle_shuffled(
                self.indices_by_stratum[stratum],
                target_length=self.num_batches * quota,
                rng=rng,
            )

        for batch_index in range(self.num_batches):
            global_batch: List[int] = []
            for stratum in self.strata:
                quota = self.quota_by_stratum[stratum]
                global_start = batch_index * quota
                global_batch.extend(
                    expanded_by_stratum[stratum][
                        global_start : global_start + quota
                    ]
                )
            random.Random(
                self.seed + self.epoch * 1_000_003 + batch_index * 97
            ).shuffle(global_batch)
            rank_start = self.rank * self.batch_size
            rank_end = rank_start + self.batch_size
            local_batch = global_batch[rank_start:rank_end]
            if len(local_batch) != self.batch_size:
                raise AssertionError(
                    f"Balanced sampler produced {len(local_batch)} rows, expected {self.batch_size}."
                )
            yield local_batch

    def __len__(self) -> int:
        return self.num_batches
