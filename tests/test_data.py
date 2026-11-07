from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import digest, load, plain, question_id
from data import native_processor, row_summary


def test_native_preprocessor_keeps_labels_outside_actor_prompt():
    row = pd.Series({'question': 'What is the capital of France?', 'data_source': 'nq',
        'reward_model': {'style': 'rule', 'ground_truth': {'target': ['Paris']}},
        'golden_answers': ['Paris'], 'ability': 'fact-reasoning', 'metadata': None})
    result = native_processor()(row, 'test', 11)
    assert result['env_kwargs'] == {'ground_truth': {'target': ['Paris']},
        'question': row['question'], 'data_source': 'nq'}
    assert result['prompt'] == [{'role': 'system', 'content': 'You are a helpful and harmless assistant.'},
                               {'role': 'user', 'content': row['question']}]
    assert 'Paris' not in str(result['prompt'])
    assert result['extra_info']['index'] == 11 and result['extra_info']['split'] == 'test'


def test_raw_ground_truth_fallback_matches_original_function():
    row = pd.Series({'question': 'Question?', 'golden_answers': ['a'], 'data_source': 'nq'})
    result = native_processor()(row, 'train', 0)
    assert result['env_kwargs']['ground_truth'] == ['a']


def test_identity_is_independent_of_numpy_serialization():
    raw = {'question': 'Q', 'ground_truth': {'target': np.array(['A', 'B'])}, 'data_source': 'nq'}
    assert question_id(raw) == question_id(plain(raw))
    changed = dict(raw, question='Other question')
    assert question_id(raw) != question_id(changed)


def test_summary_preserves_order_and_duplicates():
    a = {'question': 'A', 'ground_truth': {'target': ['x']}, 'data_source': 'nq'}
    b = {'question': 'B', 'ground_truth': {'target': ['y']}, 'data_source': 'hotpotqa'}
    frame = pd.DataFrame({'env_kwargs': [a, a, b]})
    value = row_summary(frame)
    assert value['count'] == 3 and value['source_counts'] == {'nq': 2, 'hotpotqa': 1}
    assert value['question_ids'][0] == value['question_ids'][1]
    assert value['ordered_question_ids_sha256'] != row_summary(frame.iloc[::-1])['ordered_question_ids_sha256']


def test_official_dataset_size_is_not_validation_batch_size():
    lock = load('data.json')
    assert lock['splits']['train']['rows'] == 169615
    assert lock['splits']['test']['rows'] == 51713
    assert len(lock['splits']['test']['source_counts']) == 7
    assert sum(lock['splits']['test']['source_counts'].values()) == 51713
    assert load('evaluation.json')['batch_size'] == 512
    assert (51713 + 511) // 512 == 102 and 51713 % 512 == 1
