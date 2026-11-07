import ast
import importlib.util
from pathlib import Path
import re
import string
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common import UPSTREAM


def scorer():
    source = UPSTREAM/'agent_system/environments/env_package/search/third_party/skyrl_gym/envs/search/utils.py'
    scope = {'re': re, 'string': string}
    functions = [n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), scope)
    return scope['compute_score']


@pytest.mark.parametrize('response,target,expected', [
    ('<answer>The Eiffel Tower!</answer>', ['Eiffel Tower'], 1.),
    ('<answer>PARIS</answer>', ['Paris', 'City of Paris'], 1.),
    ('<answer>Paris is in France</answer>', ['Paris'], 0.),
    ('Paris', ['Paris'], 0.),
    ('<answer>London</answer>', ['Paris'], 0.),
    ('<answer>London</answer><answer>Paris</answer>', ['Paris'], 1.),
])
def test_exact_native_normalized_em(response, target, expected):
    assert scorer()(response, {'target': target}) == expected


def test_original_parser_extracts_native_action_without_new_format_gate():
    path = UPSTREAM/'agent_system/environments/env_package/search/projection.py'
    spec = importlib.util.spec_from_file_location('native_search_projection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions, valid = module.search_projection(['<think>Find evidence.</think><search>query</search>',
        '<search>query</search><answer>answer</answer>'])
    assert actions == ['<search>query</search>', '<search>query</search>']
    assert valid == [1, 0]  # Original executor still extracts an invalid mixed response.


def test_actual_search_env_http_information_and_terminal_reward():
    from env_smoke import smoke
    value = smoke()
    assert value['status'] == 'passed' and value['native_http_search']
    assert not value['actor_model_used'] and not value['e5_faiss_tested']
