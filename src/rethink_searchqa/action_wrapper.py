"""Isolated common-action-shell contrast; original Search-QA files stay frozen.

Keep each original row intact and materialize the derived prompt/target under
``wrapper_ablation``.  Train and inference call the same prompt renderer.  The
tool adapter reverses only the shell, never rewrites query/answer content.
"""
import re
from types import SimpleNamespace

from .history_display import NATIVE, actor_input as native_input
from .protocol import MARKER, digest, native_action, parse_response, strip_termination

STYLE = "common_action_wrapper_v1"
FIELD = "wrapper_ablation"
NATIVE_TAIL = (
    "After completing your reasoning, choose only one of the following actions (do not perform both):\n"
    "(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.\n"
    "(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>."
)
WRAPPED_TAIL = (
    "After completing your reasoning, choose only one of the following actions (do not perform both):\n"
    "(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <action>search[ your query ]</action>.\n"
    "(2) If you have enough knowledge to answer the question confidently, provide your final answer within <action>answer[ your answer ]</action>, without detailed illustrations. For example, <action>answer[Beijing]</action>."
)


def render_prompt(prompt):
    """Replace the exact final instruction, not tags in questions/documents."""
    if not isinstance(prompt, str):
        raise ValueError("Observation must be text")
    body = prompt.rstrip()
    if not body.endswith(NATIVE_TAIL):
        raise ValueError("Expected the pinned native output-instruction tail")
    return body[:-len(NATIVE_TAIL)] + WRAPPED_TAIL + prompt[len(body):]


def wrap_action(text):
    parsed = native_action(text)
    kind = parsed["kind"]
    body = text[len(kind) + 2:-(len(kind) + 3)]
    if "<action>" in body or "</action>" in body:
        raise ValueError("Nested action shell in a native body")
    return f"<action>{kind}[{body}]</action>"


def unwrap_action(text):
    match = re.fullmatch(r"<action>(search|answer)\[([\s\S]*)\]</action>", text)
    if not match or "<action>" in match[2] or "</action>" in match[2]:
        raise ValueError("Require exactly one wrapped search/answer command")
    result = f"<{match[1]}>{match[2]}</{match[1]}>"
    native_action(result)
    return result


def actor_input(tokenizer, native_observation, memory=None):
    view = native_input(tokenizer, render_prompt(native_observation), memory, NATIVE)
    return dict(view, output_protocol=STYLE)


def input_for_row(row, tokenizer):
    return actor_input(tokenizer, row["prompt"], row["native_state_before"]["memory"])


def payload(row, tokenizer):
    source = {k: v for k, v in row.items() if k != FIELD}
    target = wrap_action(row["supervision_action"])
    if unwrap_action(target) != row["supervision_action"]:
        raise ValueError("Action content did not round-trip byte for byte")
    marker = MARKER if row["route"] == "bad" else ""
    if row["response"] != marker + row["supervision_action"]:
        raise ValueError("Unexpected original supervised suffix")
    return {"schema": STYLE, "source_row_sha256": digest(source),
            "actor_input": input_for_row(row, tokenizer),
            "assistant_prefix": row["assistant_prefix"],
            "response": marker + target, "supervision_action": target}


def materialize(row, tokenizer):
    if FIELD in row:
        raise ValueError("Never apply the wrapper transformation twice")
    return dict(row, **{FIELD: payload(row, tokenizer)})


def encode_row(row, tokenizer, max_length=8192):
    # The canonical validator still proves every original label/action binding.
    from .ablation_data import encode_row as native_encode
    original = native_encode(row, tokenizer, max_length)
    expected = payload(row, tokenizer)
    if row.get(FIELD) != expected:
        raise ValueError("Derived prompt/target differs from its immutable source")
    view = expected["actor_input"]
    prefix = view["visible_H"] + expected["assistant_prefix"]
    suffix = expected["response"] + tokenizer.eos_token
    p = tokenizer.encode(prefix, add_special_tokens=False)
    s = tokenizer.encode(suffix, add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    marker = tokenizer.convert_tokens_to_ids(MARKER)
    if (not p or not s or s[-1] != tokenizer.eos_token_id
            or (s[0] == marker) != bool(original["route_label"])):
        raise ValueError("Incorrect first supervised token or missing terminal EOS")
    if p[:len(view["prompt_ids"])] != view["prompt_ids"]:
        raise ValueError("SFT and free-generation H token prefixes differ")
    if decode(p + s) != prefix + suffix or tokenizer.encode(prefix + suffix, add_special_tokens=False) != p + s:
        raise ValueError("Noncanonical/merged reasoning-to-action boundary")
    response_tokens = len(p) - len(view["prompt_ids"]) + len(s)
    if len(p) + len(s) > max_length or response_tokens > 512:
        raise ValueError(f"Unchanged sequence/response budget exceeded: {row['audit_id']} ({response_tokens})")
    return dict(original, input_ids=p+s, suffix_start=len(p), suffix_tokens=len(s),
                native_prefix_preserved=False, Good_sampled_action_ids_preserved=False,
                Good_original_action_text_preserved=False,
                underlying_action_text_preserved=True, output_protocol=STYLE,
                response_tokens=response_tokens, native_suffix_tokens=original["suffix_tokens"],
                actor_input_sha256=digest(view))


def adapt_core(core):
    values = {k: getattr(core, k) for k in dir(core) if not k.startswith("__")}
    return SimpleNamespace(**dict(values, encode_row=encode_row, generation_metrics=generation_metrics))


def parsed_response(text):
    clean = strip_termination(text)
    result = {"strict_protocol": False, "marker_present": MARKER in clean,
              "marker_at_valid_boundary": False, "reasoning": None, "action": None}
    # Retain native validation of the entire think/marker/action sequence.
    match = re.search(r"<action>(?:search|answer)\[[\s\S]*\]</action>$", clean)
    if not match or clean.count("<action>") != 1 or clean.count("</action>") != 1:
        return result
    try:
        raw = unwrap_action(match[0])
    except ValueError:
        return result
    native = parse_response(clean[:match.start()] + raw)
    if native["strict_protocol"]:
        native["action"] = dict(native["action"], wrapped_text=match[0])
        return native
    return result


def to_native_response(text):
    """Decode one action shell; do not introduce a new reasoning requirement.

    Strict reasoning-format metrics are separate from tool execution.  A direct
    action remains directly executable, just as with native Search-QA.
    """
    clean = strip_termination(text)
    match = re.search(r"<action>(?:search|answer)\[[\s\S]*\]</action>", clean)
    if not match or clean.count("<action>") != 1 or clean.count("</action>") != 1:
        raise ValueError("Require one action shell for conversion")
    return clean[:match.start()] + unwrap_action(match[0]) + clean[match.end():]


def generation_metrics(text, ids, marker_id, eos_ids):
    parsed = parsed_response(text)
    return {"reasoning_open": "<think>" in text, "reasoning_closed": "</think>" in text,
            "strict_protocol": parsed["strict_protocol"], "marker_token_count": ids.count(marker_id),
            "marker_present": marker_id in ids,
            "marker_text_agrees": (marker_id in ids) == parsed["marker_present"],
            "marker_at_valid_boundary": parsed["marker_at_valid_boundary"],
            "parsed_action": parsed["action"]["wrapped_text"] if parsed["action"] else None,
            "native_action": parsed["action"]["text"] if parsed["action"] else None,
            "terminated_with_eos": bool(ids and ids[-1] in eos_ids),
            "length_truncated": bool(len(ids) >= 512 and ids[-1] not in eos_ids)}
