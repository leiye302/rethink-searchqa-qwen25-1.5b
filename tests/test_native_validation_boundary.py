"""Execute the pinned native validation method, not an invented input schema."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import question_id
from run import config


def wrapper_from_pinned_validation(monkeypatch, torch):
    class Proto:
        def __init__(self, batch, non_tensor_batch):
            self.batch, self.non_tensor_batch = batch, non_tensor_batch

        @classmethod
        def from_single_dict(cls, values):
            return cls({k: v for k, v in values.items() if torch.is_tensor(v)},
                       {k: v for k, v in values.items() if not torch.is_tensor(v)})

        def repeat(self, repeat_times, interleave):
            assert repeat_times == 1 and interleave
            return self

        def pop(self, batch_keys, non_tensor_batch_keys):
            return Proto({k: self.batch.pop(k) for k in batch_keys},
                         {k: self.non_tensor_batch.pop(k) for k in non_tensor_batch_keys})

    source = ROOT / '.deps/sdar/verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    method = next(n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == '_validate')
    namespace = {'np': np, 'torch': torch, 'DataProto': Proto}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    module = ModuleType('verl.trainer.ppo.skillsd_ray_trainer')
    module.SkillSDRayTrainer = type('LockedNative', (), {'_validate': namespace['_validate']})
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.skillsd_ray_trainer', module)
    spec = importlib.util.spec_from_file_location('actual_native_validation_fixture', ROOT / 'src/grpo_eval.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    return wrapper, Proto


@pytest.mark.parametrize('scope', ['subset', 'full'])
@pytest.mark.parametrize('mutation', ['none', 'reordered', 'duplicate', 'missing'])
def test_native_step_zero_pop_has_no_index_and_preserves_question_identity(tmp_path, monkeypatch, scope, mutation):
    torch = pytest.importorskip('torch')
    wrapper, Proto = wrapper_from_pinned_validation(monkeypatch, torch)
    inputs = [dict(question='Q' + str(i), ground_truth={'target': ['A']},
                   data_source='nq' if i < 2 else 'hotpotqa') for i in range(3)]
    expected = {'count': 3, 'question_ids': [question_id(x) for x in inputs],
                'source_counts': {'nq': 2, 'hotpotqa': 1}}
    full = tmp_path / 'full.json'
    subset = tmp_path / 'subset.json'
    full.write_text(json.dumps({'files': {'test': {'rows': expected}}}), encoding='utf-8')
    subset.write_text(json.dumps({'rows': expected, 'source_test_indices': [10, 20, 30]}), encoding='utf-8')
    monkeypatch.setenv('GRPO_RUN_DIR', str(tmp_path / 'run'))
    monkeypatch.setenv('QA_DATA_RECEIPT', str(full))
    monkeypatch.setenv('QA_SUBSET_RECEIPT', str(subset))
    monkeypatch.setenv('QA_EVAL_MODE', 'auto' if scope == 'subset' else 'full')

    actual = list(inputs)
    if mutation == 'reordered':
        actual[0], actual[1] = actual[1], actual[0]
    elif mutation == 'duplicate':
        actual[1] = actual[0]
    elif mutation == 'missing':
        actual.pop()

    batches = []
    for offset in range(0, len(actual), 2):
        items = actual[offset:offset + 2]
        count = len(items)
        batches.append({
            'input_ids': torch.ones((count, 2), dtype=torch.long),
            'attention_mask': torch.ones((count, 2), dtype=torch.long),
            'position_ids': torch.zeros((count, 2), dtype=torch.long),
            'raw_prompt_ids': np.array([[1, 1]] * count, dtype=object),
            'data_source': np.array([x['data_source'] for x in items], dtype=object),
            'env_kwargs': np.array(items, dtype=object),
            # The real dataset has this, but native _validate.pop leaves it behind.
            'index': np.arange(offset, offset + count),
        })
    calls = []

    def collect(gen_batch, **kwargs):
        assert 'index' not in gen_batch.non_tensor_batch
        items = gen_batch.non_tensor_batch.pop('env_kwargs')
        local = [0, 0, 1] if len(items) == 2 else [0]
        calls.append(len(items))
        scores = [float(items[i]['question'] != 'Q1') for i in local]
        return Proto({'responses': torch.ones((len(local), 2), dtype=torch.long),
                      'attention_mask': torch.ones((len(local), 4), dtype=torch.long)}, {
            'index': np.array(local),
            'data_source': np.array([items[i]['data_source'] for i in local]),
            'episode_rewards': np.array(scores),
            'traj_uid': np.array([items[i]['question'] for i in local]),
            'tool_callings': np.array([0] * len(local)),
            'turn_step': np.array([0, 1, 0] if len(items) == 2 else [0]),
            'is_action_valid': np.array([True] * len(local)),
        })

    trainer = wrapper.SearchTrainer()
    trainer.global_steps = 0
    trainer.config = config('3b', tmp_path / 'model', tmp_path / 'run', evaluate=True)
    trainer.val_dataset, trainer.val_dataloader = object(), batches
    trainer._searchqa_evaluation_loaders = {scope: (trainer.val_dataset, batches)}
    trainer.traj_collector = SimpleNamespace(multi_turn_loop=collect)
    trainer.actor_rollout_wg, trainer.val_envs = object(), object()
    trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0,
                                        decode=lambda ids, **kwargs: '<answer>A</answer>')
    trainer.val_reward_fn = lambda batch, **kwargs: {
        'reward_tensor': torch.tensor([[0., score] for score in batch.non_tensor_batch['episode_rewards']])}
    trainer._maybe_log_val_generations = lambda **kwargs: None

    if mutation != 'none':
        with pytest.raises(RuntimeError, match='fixed|traverse|missed'):
            trainer._validate()
        assert not list((tmp_path / 'run/evaluation').glob('update_*/attempt-*/summary.json'))
        assert trainer.traj_collector.multi_turn_loop is collect
        return
    metrics = trainer._validate()
    assert calls == [2, 1]  # Includes the final partial batch.
    assert metrics['val/nq/test_score'] == pytest.approx(2 / 3)
    assert metrics['val/hotpotqa/test_score'] == 1.
    assert metrics['eval/' + scope + '/question_micro_em'] == pytest.approx(2 / 3)
    summaries = list((tmp_path / 'run/evaluation').glob('update_000/attempt-*/summary.json'))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding='utf-8'))
    assert summary['questions'] == 3 and summary['correct'] == 2 and summary['scope'] == scope
    assert trainer.traj_collector.multi_turn_loop is collect
