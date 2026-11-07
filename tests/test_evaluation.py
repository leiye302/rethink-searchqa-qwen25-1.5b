import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import digest, question_id
from evaluation_contract import assert_protocol, summarize_questions
from run import config


def test_question_aggregation_does_not_weight_long_trajectories():
    rows = [dict(index=i, question_id=str(i), data_source=source, episode_reward=score)
            for i, (source, score) in enumerate([('nq', 1.), ('nq', 0.), ('hotpotqa', 1.)])]
    result = summarize_questions(rows, ['0', '1', '2'], {'nq': 2, 'hotpotqa': 1})
    assert result['question_micro_em'] == pytest.approx(2/3)
    assert result['dataset_macro_em'] == .75
    assert result['by_dataset']['nq']['em'] == .5


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'replaced', 'nonbinary'])
def test_incomplete_or_changed_test_cannot_be_reported_as_full(mode):
    row = {'index': 0, 'question_id': 'zero', 'data_source': 'nq', 'episode_reward': 1.}
    rows = [row]
    if mode == 'missing': rows = []
    if mode == 'duplicate': rows = [row, row]
    if mode == 'replaced': rows = [dict(row, question_id='different')]
    if mode == 'nonbinary': rows = [dict(row, episode_reward=.5)]
    with pytest.raises(RuntimeError):
        summarize_questions(rows, ['zero'], {'nq': 1})


def test_sampling_drift_is_not_accepted(tmp_path):
    cfg = config('3b', tmp_path/'model', tmp_path/'run')
    cfg.actor_rollout_ref.rollout.val_kwargs.do_sample = True
    with pytest.raises(RuntimeError):
        assert_protocol(cfg)


def test_real_wrapper_handles_native_local_index_reset_and_preserves_metrics(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    inputs = [dict(question='Q'+str(i), ground_truth={'target':['A']},
                   data_source='nq' if i < 2 else 'hotpotqa') for i in range(3)]
    receipt = {'files': {'test': {'rows': {'question_ids': [question_id(x) for x in inputs],
                                         'count': 3,
                                         'source_counts': {'nq': 2, 'hotpotqa': 1}}}}}
    receipt_path = tmp_path/'data_receipt.json'
    receipt_path.write_text(json.dumps(receipt))
    native_metrics = {'val/success_rate': .75, 'val/nq/test_score': 2/3}

    class Native:
        def _validate(self):
            for offset, count in [(0, 2), (2, 1)]:
                gen = SimpleNamespace(non_tensor_batch={'env_kwargs': np.array(inputs[offset:offset+count], dtype=object),
                    'data_source': np.array([x['data_source'] for x in inputs[offset:offset+count]])})
                self.traj_collector.multi_turn_loop(gen_batch=gen, is_train=False)
            return native_metrics

    module = ModuleType('verl.trainer.ppo.skillsd_ray_trainer')
    module.SkillSDRayTrainer = Native
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.skillsd_ray_trainer', module)
    spec = importlib.util.spec_from_file_location('tested_search_eval', ROOT/'src/grpo_eval.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    monkeypatch.setenv('GRPO_RUN_DIR', str(tmp_path/'run'))
    monkeypatch.setenv('QA_DATA_RECEIPT', str(receipt_path))
    calls = []

    def collect(gen_batch, **kwargs):
        # Match native behavior: env_kwargs is consumed; output index is local.
        count = len(gen_batch.non_tensor_batch.pop('env_kwargs'))
        offset = sum(calls)
        local = [0, 0, 1] if count == 2 else [0]
        calls.append(count)
        nt = {'index': local,
            'traj_uid': ['t'+str(offset+i) for i in local],
            'episode_rewards': [1. if offset+i != 1 else 0. for i in local],
            'data_source': [inputs[offset+i]['data_source'] for i in local],
            'turn_step': [0, 1, 0] if count == 2 else [0],
            'is_action_valid': [True]*len(local)}
        return SimpleNamespace(batch={'responses': torch.ones((len(local), 2), dtype=torch.long),
            'attention_mask': torch.ones((len(local), 4), dtype=torch.long)}, non_tensor_batch=nt)

    trainer = wrapper.SearchTrainer()
    trainer.global_steps = 150
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(val_only=True))
    trainer.val_dataset = object()
    trainer.val_dataloader = object()
    trainer._searchqa_evaluation_loaders = {'full': (trainer.val_dataset, trainer.val_dataloader)}
    trainer.traj_collector = SimpleNamespace(multi_turn_loop=collect)
    trainer.tokenizer = SimpleNamespace(decode=lambda ids,**kwargs:'<answer>A</answer>')
    reported = trainer._validate()
    assert all(reported[key] == value for key, value in native_metrics.items())
    assert reported['eval/full/question_micro_em'] == pytest.approx(2/3)
    assert trainer.traj_collector.multi_turn_loop is collect and calls == [2, 1]
    summaries = list((tmp_path/'run/evaluation').glob('update_150/attempt-*/summary.json'))
    assert len(summaries) == 1
    value = json.loads(summaries[0].read_text(encoding='utf-8'))
    assert value['questions'] == 3 and value['correct'] == 2
    assert value['question_micro_em'] == pytest.approx(2/3)
    assert value['native_metrics_unchanged'] == native_metrics
    assert value['scope'] == 'full' and value['wall_seconds'] >= 0
