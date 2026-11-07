"""Registration and validation helpers for the Routed-OPSD control token.

``<RETHINK>`` must be a real special token, but its numeric id is deliberately
not part of the experiment configuration.  Qwen tokenizers and model heads can
have different apparent vocabulary sizes, so guessing an id (or blindly
calling ``resize_token_embeddings(len(tokenizer))``) can either collide with a
reserved token or shrink valid Qwen embedding rows.

The helpers in this module always resolve the id from the tokenizer and only
grow model embeddings when the resolved id would otherwise be out of range.
They never shrink an embedding matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RETHINK_TOKEN = "<RETHINK>"


class RethinkTokenError(ValueError):
    """Raised when the control token contract is not satisfied."""


@dataclass(frozen=True)
class RethinkTokenState:
    """Validated tokenizer/model state after registration or reload."""

    token: str
    token_id: int
    tokenizer_size: int
    input_embedding_rows: int | None
    output_embedding_rows: int | None
    added: bool = False
    resized: bool = False


def _encode_without_special_tokens(tokenizer: Any, token: str) -> list[int]:
    try:
        encoded = tokenizer.encode(token, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(token)

    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise RethinkTokenError(f"{token!r} produced a batched encoding with {len(encoded)} rows")
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)):
        raise RethinkTokenError(f"{token!r} produced an unsupported encoding: {type(encoded).__name__}")
    return [int(item) for item in encoded]


def resolve_single_token_id(tokenizer: Any, token: str = RETHINK_TOKEN) -> int:
    """Resolve ``token`` and fail unless it is exactly one non-unknown token.

    This is the runtime-facing API used by Stage 2.  It intentionally performs
    no registration: training and inference must load the already-persisted
    tokenizer from the checkpoint.
    """

    token_ids = _encode_without_special_tokens(tokenizer, token)
    if len(token_ids) != 1:
        raise RethinkTokenError(
            f"{token!r} must encode to exactly one token, but encoded to {token_ids}. "
            "Prepare the checkpoint with scripts/rethink_opsd/prepare_rethink_checkpoint.py."
        )

    token_id = token_ids[0]
    if token_id < 0:
        raise RethinkTokenError(f"{token!r} resolved to a negative token id: {token_id}")

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)
    if unk_token_id is not None and token_id == int(unk_token_id) and token != unk_token:
        raise RethinkTokenError(f"{token!r} resolved to unk_token_id={unk_token_id}")

    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        resolved_text = convert(token_id)
        if isinstance(resolved_text, (list, tuple)):
            resolved_text = resolved_text[0] if len(resolved_text) == 1 else resolved_text
        if resolved_text != token:
            raise RethinkTokenError(
                f"token id {token_id} maps back to {resolved_text!r}, not the requested control token {token!r}"
            )
    return token_id


def validate_rethink_token_absent(
    tokenizer: Any,
    token: str = RETHINK_TOKEN,
) -> None:
    """Fail when ``token`` is registered as a usable single control token.

    The paper SEED baseline intentionally uses the pre-method tokenizer.  Its
    launch preflight uses this inverse contract so a checkpoint that already
    exposes the method token cannot silently enter the baseline group.
    """

    try:
        token_id = resolve_single_token_id(tokenizer, token)
    except RethinkTokenError:
        return
    raise RethinkTokenError(
        f"paper SEED baseline must not register {token!r}, but it resolves to token id {token_id}"
    )


def _special_tokens(tokenizer: Any) -> set[str]:
    values: set[str] = set()
    for attribute in ("all_special_tokens", "additional_special_tokens"):
        entries = getattr(tokenizer, attribute, None) or []
        values.update(str(entry) for entry in entries)
    return values


def _tokenizer_size(tokenizer: Any, token_id: int) -> int:
    try:
        size = int(len(tokenizer))
    except (TypeError, AttributeError):
        size = int(getattr(tokenizer, "vocab_size", token_id + 1))
    return max(size, token_id + 1)


def _embedding_rows(embedding: Any, *, name: str) -> int:
    if embedding is None:
        raise RethinkTokenError(f"model does not expose {name} embeddings")
    rows = getattr(embedding, "num_embeddings", None)
    if rows is None:
        weight = getattr(embedding, "weight", None)
        shape = getattr(weight, "shape", None)
        if not shape:
            raise RethinkTokenError(f"cannot determine row count for {name} embeddings")
        rows = shape[0]
    return int(rows)


def validate_rethink_token(
    tokenizer: Any,
    model: Any | None = None,
    token: str = RETHINK_TOKEN,
) -> RethinkTokenState:
    """Validate single-token, special-token, and optional embedding invariants."""

    token_id = resolve_single_token_id(tokenizer, token)
    if token not in _special_tokens(tokenizer):
        raise RethinkTokenError(f"{token!r} is one token but is not registered as a special token")

    tokenizer_size = _tokenizer_size(tokenizer, token_id)
    input_rows: int | None = None
    output_rows: int | None = None
    if model is not None:
        input_rows = _embedding_rows(model.get_input_embeddings(), name="input")
        output_rows = _embedding_rows(model.get_output_embeddings(), name="output/LM-head")
        if token_id >= input_rows:
            raise RethinkTokenError(
                f"{token!r} id {token_id} is outside input embedding range [0, {input_rows})"
            )
        if token_id >= output_rows:
            raise RethinkTokenError(
                f"{token!r} id {token_id} is outside output/LM-head range [0, {output_rows})"
            )
        config_vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if config_vocab_size is not None and token_id >= int(config_vocab_size):
            raise RethinkTokenError(
                f"{token!r} id {token_id} is outside model.config.vocab_size={config_vocab_size}"
            )

    return RethinkTokenState(
        token=token,
        token_id=token_id,
        tokenizer_size=tokenizer_size,
        input_embedding_rows=input_rows,
        output_embedding_rows=output_rows,
    )


def ensure_rethink_token(
    tokenizer: Any,
    model: Any | None = None,
    token: str = RETHINK_TOKEN,
) -> RethinkTokenState:
    """Register and validate ``token``, growing embeddings only when required.

    Qwen3 normally has spare model rows after the tokenizer vocabulary.  In
    that case registration consumes a tokenizer id within the existing model
    range and this function leaves both matrices untouched.  If a different
    checkpoint has no spare row, the matrices are grown to the smallest safe
    size.  They are never resized downward.
    """

    added = False
    if token not in _special_tokens(tokenizer):
        payload = {"additional_special_tokens": [token]}
        try:
            num_added = tokenizer.add_special_tokens(payload, replace_additional_special_tokens=False)
        except TypeError:
            # Compatibility with older/custom tokenizers.  Preserve any
            # existing additional special tokens when the newer keyword is not
            # supported.
            existing = list(getattr(tokenizer, "additional_special_tokens", None) or [])
            if token not in existing:
                existing.append(token)
            num_added = tokenizer.add_special_tokens({"additional_special_tokens": existing})
        added = bool(num_added) or token in _special_tokens(tokenizer)

    token_id = resolve_single_token_id(tokenizer, token)
    required_rows = _tokenizer_size(tokenizer, token_id)
    resized = False

    if model is not None:
        input_rows = _embedding_rows(model.get_input_embeddings(), name="input")
        output_rows = _embedding_rows(model.get_output_embeddings(), name="output/LM-head")
        if required_rows > input_rows or required_rows > output_rows:
            target_rows = max(required_rows, input_rows, output_rows)
            resize = getattr(model, "resize_token_embeddings", None)
            if not callable(resize):
                raise RethinkTokenError(
                    f"model needs {target_rows} rows for {token!r}, but has no resize_token_embeddings method"
                )
            resize(target_rows)
            tie_weights = getattr(model, "tie_weights", None)
            if callable(tie_weights):
                tie_weights()
            resized = True

        # Keep config usable by Hugging Face and vLLM even for models that
        # expose preallocated embedding rows larger than len(tokenizer).
        final_input_rows = _embedding_rows(model.get_input_embeddings(), name="input")
        final_output_rows = _embedding_rows(model.get_output_embeddings(), name="output/LM-head")
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "vocab_size"):
            config.vocab_size = max(int(config.vocab_size), token_id + 1, final_input_rows, final_output_rows)

    validated = validate_rethink_token(tokenizer, model=model, token=token)
    return RethinkTokenState(
        token=validated.token,
        token_id=validated.token_id,
        tokenizer_size=validated.tokenizer_size,
        input_embedding_rows=validated.input_embedding_rows,
        output_embedding_rows=validated.output_embedding_rows,
        added=added,
        resized=resized,
    )
