"""Dataset checks only. Execution always uses SDAR's unchanged projection."""
import hashlib
import json
import re

MARKER = '<RETHINK>'
ACTION = re.compile(r'<(search|answer)>([\s\S]*?)</\1>')
TERMINATORS = re.compile(r'(?:<\|im_end\|>|<\|endoftext\|>)\s*$')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stable_question(question):
    return digest(' '.join(question.split()).casefold())


def strip_termination(text):
    # Termination is allowed only at the end, not silently removed inside R/A.
    return TERMINATORS.sub('', text.rstrip())


def native_action(text):
    if not isinstance(text, str):
        raise ValueError('Action must be text')
    match = ACTION.fullmatch(text)
    if not match or not match[2].strip() or re.search(
            r'</?(?:think|search|answer|information)>|<RETHINK>|<\|', match[2]):
        raise ValueError('Require exactly one nonempty native search OR answer action')
    return {'kind': match[1], 'content': match[2].strip(), 'text': text,
            'executed': f'<{match[1]}>{match[2].strip()}</{match[1]}>'}


def parse_response(text):
    clean = strip_termination(text)
    match = re.fullmatch(r'\s*(<think>([\s\S]*?)</think>)(\s*)(<RETHINK>)?(<(?:search|answer)>[\s\S]*)', clean)
    result = {'strict_protocol': False, 'marker_present': MARKER in clean,
              'marker_at_valid_boundary': False, 'reasoning': None, 'action': None}
    if not match or not match[2].strip():
        return result
    try:
        action = native_action(match[5])
    except ValueError:
        return result
    if clean.count('<think>') != 1 or clean.count('</think>') != 1 or clean.count(MARKER) != bool(match[4]):
        return result
    result.update(strict_protocol=True, marker_at_valid_boundary=bool(match[4]),
                  reasoning=match[2], action=action,
                  assistant_prefix=clean[:match.start(4) if match[4] else match.start(5)])
    return result


def split_natural(text):
    parsed = parse_response(text)
    if not parsed['strict_protocol'] or parsed['marker_present']:
        raise ValueError('Require a complete, unmodified natural R/action response without marker')
    return parsed['assistant_prefix'], parsed['action']['text']


def blind_payload(row):
    # Explicit allowlist: no action, reward, gold answer, next observation,
    # teacher reasoning, label provenance, source name or full hidden history.
    prefix, _ = split_natural(row['assistant_prefix'] + row['source_executed_action'])
    reasoning = parse_response(prefix + row['source_executed_action'])['reasoning']
    return {'visible_H': row['visible_H'], 'reasoning': reasoning}


def label_binding(row):
    return digest(blind_payload(row))
