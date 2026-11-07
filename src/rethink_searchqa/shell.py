"""The wrapped Stage-1 prompt and a reversible native action adapter."""
import re
from . import action_wrapper
from .protocol import digest
from .history_display import NATIVE, actor_input as native_input


MARKER = '<RETHINK>'
MARKER_ID = 151665
STYLE = action_wrapper.STYLE
RETHINK_INTRO = (
    'If, after completing your reasoning, you identify a concrete, uncorrected '
    'mistake in it that would mislead your next decision, write <RETHINK> once '
    'immediately after </think>, then give the corrected action. Otherwise, '
    'give the action directly after </think>. <RETHINK> is an optional marker, '
    'not an action.\n'
)
INTRO_DIGEST = '27d762cbccb5d2672d8dbe9b20a583bdade9781d88fcdb95d90810882c0c8d16'
if digest(RETHINK_INTRO) != INTRO_DIGEST:
    raise RuntimeError('The frozen Stage-1 RETHINK instruction differs')


def render(observation):
    """Preserve the exact original history, including trailing whitespace."""
    body = observation.rstrip()
    tail = action_wrapper.NATIVE_TAIL
    if not body.endswith(tail) or RETHINK_INTRO in body:
        raise ValueError('Unexpected native Search-QA prompt boundary')
    return body[:-len(tail)] + RETHINK_INTRO + action_wrapper.WRAPPED_TAIL + observation[len(body):]


def actor_input(tokenizer, observation, memory=None):
    view = native_input(tokenizer, render(observation), memory, NATIVE)
    return dict(view, output_protocol=STYLE)


def native_response(text):
    """Match the Stage-1 real benchmark, not a new format reward or gate."""
    try:
        return action_wrapper.to_native_response(text)
    except ValueError:
        def convert(match):
            try:
                return action_wrapper.unwrap_action(match[0])
            except ValueError:
                return match[0]
        return re.sub(r'<action>(?:search|answer)\[(?:(?!</?action>)[\s\S])*?\]</action>',
                      convert, text)


def wire_environment(env, *, capture_hints):
    """Only render Actor prompts and translate the shell before native parsing."""
    if hasattr(env, '_rethink_capture_hints'):
        raise ValueError('Environment is already wired')
    build, project = env.build_text_obs, env.projection_f
    env.build_text_obs = lambda *a, **kw: [render(x) for x in build(*a, **kw)]
    env.projection_f = lambda responses: project([native_response(x) for x in responses])
    env._rethink_capture_hints = bool(capture_hints)
    return env


def assert_token(tokenizer):
    if (tokenizer.encode(MARKER, add_special_tokens=False) != [MARKER_ID]
            or tokenizer.convert_ids_to_tokens(MARKER_ID) != MARKER):
        raise ValueError('Expected the registered one-token RETHINK Stage-1 Actor')
