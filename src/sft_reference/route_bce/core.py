from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from legacy.rethink_token import ensure_rethink_token
from legacy.stage1_training import TaskBalancedDistributedBatchSampler

MARKER = "<RETHINK>"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, payload):
    """Atomic generated report; callers own the exact destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".next")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def register_token(tokenizer, model=None):
    state = ensure_rethink_token(tokenizer, model=model)
    # Qwen2.5's preallocated vocabulary has room. A resize here would mean a
    # different parent and is rejected for this frozen experiment.
    if model is not None and state.resized:
        raise ValueError("Unexpected model resize for the pinned Qwen2.5 parent")
    return asdict(state)


def encode_row(row, tokenizer, max_length=8192):
    """Replicate legacy SFTDataset's two-piece tokenization and shifted mask."""
    if row["route"] not in ("good", "bad"):
        raise ValueError("Uncertain/unreviewed rows cannot enter training")
    prefix = row["assistant_prefix"]
    if not prefix.startswith("<think>") or not prefix.rstrip().endswith("</think>"):
        raise ValueError("Invalid R boundary")
    label = int(row["route"] == "bad")
    expected = (MARKER if label else "") + f"<action>{row['supervision_action']}</action>"
    if row["response"] != expected or row["response"].count(MARKER) != label:
        raise ValueError("Route/action target mismatch")
    if not label and row["supervision_action"] != row["source_executed_action"]:
        raise ValueError("Good action must remain the original natural sample")
    if label and row["supervision_action"] not in row["admissible_actions_before"]:
        raise ValueError("Bad action lacks stored admissibility evidence")
    h = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                      tokenize=False, add_generation_prompt=True)
    # No analyzer text/label/Oracle metadata reaches the model.
    prefix_text = h + prefix
    suffix_text = row["response"] + tokenizer.eos_token
    p = tokenizer.encode(prefix_text, add_special_tokens=False)
    s = tokenizer.encode(suffix_text, add_special_tokens=False)
    marker_id = tokenizer.convert_tokens_to_ids(MARKER)
    action_first = tokenizer.encode("<action>", add_special_tokens=False)[0]
    if not p or not s or s[0] != (marker_id if label else action_first):
        raise ValueError("First supervised token must be marker/action, not whitespace")
    if s[-1] != tokenizer.eos_token_id:
        raise ValueError("EOS is missing from supervision")
    if len(p) + len(s) > max_length:
        raise ValueError("truncation=error: overlength row " + row["audit_id"])
    return {"input_ids": p + s, "suffix_start": len(p), "route_label": label,
            "joint_tokenization_matches": tokenizer.encode(prefix_text + suffix_text, add_special_tokens=False) == p + s,
            "suffix_tokens": len(s), "audit_id": row["audit_id"]}


def losses(logits, input_ids, suffix_start, route_label, marker_id, route_lambda=0.1):
    """One unpadded row: sample-mean suffix CE + one full-vocabulary BCE.

    suffix_start is the index of the first TARGET token. Both losses read its
    predictive logits at suffix_start - 1, never after consuming the marker.
    """
    if logits.ndim != 2 or input_ids.ndim != 1 or logits.shape[0] != input_ids.numel():
        raise ValueError("Expect unpadded [T,V] logits and [T] input_ids")
    if not 0 < suffix_start < input_ids.numel() or route_label not in (0, 1):
        raise ValueError("Invalid decision boundary/label")
    if not 0 <= marker_id < logits.shape[-1] or route_lambda < 0:
        raise ValueError("Invalid marker/lambda")
    if (int(input_ids[suffix_start]) == marker_id) != bool(route_label):
        raise ValueError("Marker target and route label disagree")
    # Cast only the short supervised suffix, not the entire H/R vocabulary
    # cube. Context is present and differentiable but receives no token CE.
    suffix_logits = logits[suffix_start - 1:-1].float()
    targets = input_ids[suffix_start:]
    ce = F.cross_entropy(suffix_logits, targets, reduction="mean")
    z = suffix_logits[0]
    not_marker_lse = torch.logsumexp(torch.cat((z[:marker_id], z[marker_id + 1:])), dim=0)
    binary_logit = z[marker_id] - not_marker_lse
    bce = F.binary_cross_entropy_with_logits(binary_logit, z.new_tensor(float(route_label)))
    return {"loss": ce + route_lambda * bce, "suffix_ce": ce, "route_bce": bce,
            "marker_probability": binary_logit.sigmoid(), "binary_logit": binary_logit}


def batches(rows, seed=0, epoch=0):
    """Unmodified legacy global sampler; exact quotas use each row once."""
    sampler = TaskBalancedDistributedBatchSampler(
        task_kinds=["actor_route"] * len(rows), task_kind_values=["actor_route"],
        routes=[r["route"] for r in rows], route_values=["good", "bad"],
        route_task_kind="actor_route", route_batch_counts={"good": 5, "bad": 3},
        epoch_anchor_route="good", batch_size=8, num_replicas=1, rank=0, seed=seed)
    sampler.set_epoch(epoch)
    order = list(sampler)
    flattened = [i for b in order for i in b]
    if len(flattened) != len(rows) or len(set(flattened)) != len(rows):
        raise ValueError("Frozen one-epoch recipe forbids cycling or dropping rows")
    return order


def generation_metrics(text, ids, marker_id, eos_ids):
    # Keep special tokens when decoding. Remove only termination/pad protocol
    # tokens for structural validation; never strip RETHINK.
    clean = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    pattern = r"<think>\s*(?P<r>[\s\S]*?\S)\s*</think>\s*(?P<m><RETHINK>)?\s*<action>(?P<a>[^<>\r\n]+)</action>"
    match = re.fullmatch(pattern, clean)
    strict = (bool(match) and clean.count("<think>") == 1 and clean.count("</think>") == 1
              and clean.count("<action>") == 1 and clean.count("</action>") == 1
              and clean.count(MARKER) == int(match.group("m") is not None))
    has_marker = marker_id in ids
    action = match.group("a") if strict else None
    return {"reasoning_open": "<think>" in clean, "reasoning_closed": "</think>" in clean,
            "strict_protocol": strict, "marker_token_count": ids.count(marker_id),
            "marker_present": has_marker, "marker_text_agrees": has_marker == (MARKER in text),
            "parsed_action": action, "terminated_with_eos": bool(ids and ids[-1] in eos_ids)}
