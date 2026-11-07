"""One retrospective request per complete episode; Actor owns all target IDs."""
import hashlib
import json
import re
try:
    from .rethink_benchmark_contract import benchmark, command_allowed, hint_command, hint_instruction
except ImportError:  # existing standalone API worker / file-loader tests
    import importlib.util
    from pathlib import Path
    _spec = importlib.util.spec_from_file_location("rh_benchmark_contract", Path(__file__).with_name("rethink_benchmark_contract.py"))
    _contract = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_contract)
    benchmark, command_allowed = _contract.benchmark, _contract.command_allowed
    hint_command, hint_instruction = _contract.hint_command, _contract.hint_instruction

VERSION = "actor_only_episode_hints_glm_v2_20261015"
MODEL = "glm-5.3-flash"
INSTRUCTION = """You supply short ALFWorld hints to a privileged Teacher, not to the Actor.
The Actor has ALREADY emitted RETHINK at every requested target step. Do not judge
good/bad, approve, reject, or select targets. Return one hint for EVERY requested
step, including when its original plan/action is sensible; confirming it is fine.
You receive one complete chronological trajectory. Actor reasoning is fallible;
environment observations and feedback are factual. At target t, give a useful
NEXT command for the state immediately BEFORE action t, not for the final state.
Action t and feedback t have NOT happened at this decision point. Later feedback
may help hindsight planning, but never claim its facts were already observed,
an object already held/placed/cleaned/cooled, or a receptacle already searched at t.
Reconstruct each target state separately from steps STRICTLY BEFORE t and that
target's observation_before. A successful pickup means held until a successful
move; failed commands do not change inventory. For two-object tasks, leave the
first delivered object in place and obtain/place a DISTINCT second object. For
clean/heat/cool tasks, preserve verified transformations; do not repeat them.
Search choices are suggestions, not proof of an unseen object's location.
The compact trajectory preserves all original reasoning/actions/feedback. A
step with observation_before="=previous_feedback" has exactly the preceding
step's feedback as its observation. Command-set IDs refer to command_sets.
Each target snapshot repeats its exact current observation and allowed commands.
Choose the next command EXACTLY from that snapshot's admissible_actions. Prefer
meaningful progress over help/look or repeated already-exhausted searches.
Return JSON only with one top-level key "hints". Its value is an object mapping
each EXACT requested step ID (as a string key) directly to its hint text. Use the
output_template at the end of the input: fill its values, do not change its keys.
No arrays and no nested reasoning/quality fields. Each hint
must start with Next command and contain at most two short sentences (80 words).
No quality labels, no route decisions, no replacement reasoning/action outputs.
"""


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def group_requests(requests):
    groups = {}
    for request in requests:
        episode = {k: v for k, v in request["episode"].items() if k != "target_steps"}
        key = digest(episode)
        group = groups.setdefault(key, {"key": key, "episode": episode, "targets": {}})
        target = int(request["target"])
        if sum(int(s["step"]) == target for s in episode["steps"]) != 1:
            raise ValueError("Target missing or duplicated in trajectory")
        group["targets"].setdefault(target, []).append(request["key"])
    result = []
    for key in sorted(groups):
        group = groups[key]
        group["targets"] = {str(t): sorted(set(keys)) for t, keys in sorted(group["targets"].items())}
        result.append(group)
    return result


def messages(group):
    episode = group["episode"]
    environment = benchmark(episode.get("benchmark", "alfworld"))
    command_sets, lookup, steps = [], {}, []
    previous = None
    for step in episode["steps"]:
        commands = tuple(step["admissible_actions"])
        if commands not in lookup:
            lookup[commands] = len(command_sets)
            command_sets.append(list(commands))
        steps.append({"step": int(step["step"]),
                      "observation_before": "=previous_feedback" if previous == step["observation"] else step["observation"],
                      "command_set": lookup[commands], "reasoning": step["reasoning"],
                      "action": step["action"], "feedback": step["feedback"]})
        previous = step["feedback"]
    targets = [{"step": int(s["step"]), "observation_before": s["observation"],
                "admissible_actions": s["admissible_actions"],
                "boundary": "Before this step's action/feedback; do not use later state as current."}
               for s in episode["steps"] if str(s["step"]) in group["targets"]]
    payload = {"task": episode["task"], "command_sets": command_sets,
               "complete_chronological_trajectory": steps, "requested_targets": targets,
               "output_template": {"hints": {str(t["step"]): "Next command: 'COPY AN EXACT ALLOWED COMMAND'. Brief guidance."
                                             for t in targets}}}
    if environment != "alfworld":
        payload["benchmark"] = environment
        for key in payload["output_template"]["hints"]:
            payload["output_template"]["hints"][key] = "Next command: 'NATIVE_COMMAND'. Brief guidance."
    return [{"role": "system", "content": INSTRUCTION if environment == "alfworld" else hint_instruction(environment)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def request_body(group):
    return {"model": MODEL, "messages": messages(group), "temperature": 0,
            "reasoning_effort": "low", "max_tokens": min(16384, 1024 + 192 * len(group["targets"])),
            "response_format": {"type": "json_object"}}


def parse_hints(content, group, finish_reason):
    """Exact ID coverage; invalid commands fail only that hint, never Actor routing."""
    try:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result
        parsed = json.loads(content, object_pairs_hook=unique_object)
        rows = parsed["hints"]
        assert finish_reason == "stop" and isinstance(rows, dict)
        assert set(rows) == set(group["targets"])
    except (ValueError, TypeError, KeyError, AssertionError):
        return {key: {"payload": None, "error": "batch_schema_or_target_coverage"}
                for keys in group["targets"].values() for key in keys}
    outputs = {}
    environment = benchmark(group["episode"].get("benchmark", "alfworld"))
    for target_id, hint in rows.items():
        target = next(s for s in group["episode"]["steps"] if str(s["step"]) == target_id)
        command = re.match(r"^Next command:\s*['\"]([^'\"]+)['\"]", hint or "") if isinstance(hint, str) else None
        valid = (command is not None and command[1] in target["admissible_actions"]
                 and len(hint.split()) <= 100 and len(hint) <= 1000
                 and "[Actor-triggered local hint]" not in hint and "[End local hint]" not in hint)
        if environment != "alfworld":
            command_text = hint_command(hint) if isinstance(hint, str) else None
            valid = (command_text is not None and command_allowed(command_text, target["admissible_actions"], environment)
                     and len(hint.split()) <= 100 and len(hint) <= 1000
                     and "[Actor-triggered local hint]" not in hint and "[End local hint]" not in hint)
        record = {"payload": {"reasoning_skill": hint} if valid else None,
                  "error": None if valid else "invalid_hint_command_or_length"}
        for key in group["targets"][target_id]:
            outputs[key] = record
    return outputs
