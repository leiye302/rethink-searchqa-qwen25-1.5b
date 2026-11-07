"""One history renderer shared by frozen SFT inputs and live Actor inference.

Only display changes. Native memory, retrieval payloads and tool execution stay
unchanged. Reconstruct and verify the original history before any substitution.
"""
import json
import re

NATIVE = 'native_search_tags_v1'
PLAIN = 'plain_action_observation_v1'
STYLES = (NATIVE, PLAIN)
INTRO = ('Below is the interaction history where <search> </search> wrapped your '
         'past search queries and <information> </information> wrapped the '
         'corresponding search results returned by the external search engine. History:\n')
PLAIN_INTRO = ('Below is the interaction history of your past actions and the '
               'corresponding observations returned by the external search engine. History:\n')


def native_context(memory, history_length=4):
    recent = memory[-history_length:]
    start = len(memory)-len(recent)
    return '\n'.join(f"Step {start+j+1}:{rec['search']} {rec['information']}\n"
                     for j, rec in enumerate(recent))


def action_display(action):
    if not isinstance(action, str):
        raise ValueError('Native history action must be text')
    match = re.fullmatch(r'<(search|answer)>([\s\S]*)</\1>', action)
    if match:
        # JSON quoting is reversible even for brackets, quotes and newlines.
        return match[1]+'['+json.dumps(match[2], ensure_ascii=False)+']'
    return 'raw['+json.dumps(action, ensure_ascii=False)+']'


def observation_display(observation):
    if not isinstance(observation, str):
        raise ValueError('Native history observation must be text')
    if observation.startswith('<information>') and observation.endswith('</information>'):
        return observation[len('<information>'):-len('</information>')]
    return observation


def plain_context(memory, history_length=4):
    recent = memory[-history_length:]
    start = len(memory)-len(recent)
    return '\n'.join(
        f"Action {start+j+1}: {action_display(rec['search'])}\n"
        f"Observation {start+j+1}: {observation_display(rec['information'])}\n"
        for j, rec in enumerate(recent))


def render(native_observation, memory, style=NATIVE):
    if style not in STYLES:
        raise ValueError('Unknown history display style')
    if style == NATIVE:
        return native_observation
    if memory is None:
        raise ValueError('Plain history requires the actual native memory')
    if not memory:
        if INTRO in native_observation:
            raise ValueError('Empty memory is incompatible with a history prompt')
        return native_observation  # Initial native NO_HIS prompt stays exact.
    old = INTRO+native_context(memory)
    if native_observation.count(old) != 1:
        raise ValueError('Prompt history does not match the actual native state')
    new = PLAIN_INTRO+plain_context(memory)
    return native_observation.replace(old, new, 1)


def actor_input(tokenizer, native_observation, memory=None, style=NATIVE):
    observation = render(native_observation, memory, style)
    text = tokenizer.apply_chat_template([{'role':'user', 'content':observation}],
                                         tokenize=False, add_generation_prompt=True)
    full = tokenizer.encode(text, add_special_tokens=False)
    ids = full[-4096:]
    visible = tokenizer.decode(ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    return {'history_display_style':style, 'observation_prompt':observation,
            'prompt_ids':ids, 'full_prompt_tokens':len(full), 'visible_H':visible}


def input_for_row(row, tokenizer, style=NATIVE):
    return actor_input(tokenizer, row['prompt'],
                       row['native_state_before']['memory'], style)
