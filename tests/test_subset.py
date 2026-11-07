import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import digest, load, question_id
from evaluation_contract import validation_scope
from validation_subset import manifest, select_indices
from run import config


def test_fixed_shared_manifest_has_equal_sources_unique_ids_and_stable_order():
    value = manifest()
    assert value['questions'] == 448 and len(value['source_counts']) == 7
    assert set(value['source_counts'].values()) == {64}
    assert value['ordered_question_ids_sha256'] == '29d67a3da4b11211f8cfd1cd95c286375936c3e075a80109207c273c332b7bda'
    assert len({row['test_index'] for row in value['rows']}) == 448
    assert len({row['question_id'] for row in value['rows']}) == 448
    sources = sorted(value['source_counts'])
    assert [row['data_source'] for row in value['rows']] == sources * 64


def test_balanced_selector_excludes_training_matches_and_duplicate_question_text():
    rows = [{'question': source + str(i), 'data_source': source} for source in ['a', 'b', 'c'] for i in range(12)]
    rows += [{'question': ' Shared Question ', 'data_source': 'a'}, {'question': 'shared question', 'data_source': 'b'}]
    indices = select_indices(rows, [' A0 ', 'B0'], ['a', 'b', 'c'], per_source=5, seed=2026)
    assert indices == select_indices(rows, [' A0 ', 'B0'], ['a', 'b', 'c'], per_source=5, seed=2026)
    from collections import Counter
    assert Counter(rows[i]['data_source'] for i in indices) == {'a': 5, 'b': 5, 'c': 5}
    texts = [rows[i]['question'].casefold().strip() for i in indices]
    assert len(set(texts)) == len(texts)
    assert not {'a0', 'b0'} & set(texts)
    with pytest.raises(RuntimeError, match='Not enough'):
        select_indices(rows, [], ['a', 'b', 'c'], per_source=100)


def test_initial_and_every20_monitor_only_final150_is_full():
    schedule = load('evaluation.json')
    assert schedule['training_validation_updates'] == [0, 20, 40, 60, 80, 100, 120, 140, 150]
    assert [s for s in schedule['training_validation_updates'] if validation_scope(s) == 'full'] == [150]
    assert all(validation_scope(s) == 'subset' for s in range(0, 150, 20))
    assert validation_scope(0, 'full') == 'full'  # explicit independent final-model eval
    assert validation_scope(150, 'subset') == 'subset'
    with pytest.raises(RuntimeError): validation_scope(151)
    with pytest.raises(RuntimeError): validation_scope(0, 'unknown')


def test_independent_eval_defaults_to_full_but_training_initializes_subset(tmp_path):
    train = config('3b', tmp_path / 'model', tmp_path / 'train')
    full = config('3b', tmp_path / 'model', tmp_path / 'full', evaluate=True)
    short = config('3b', tmp_path / 'model', tmp_path / 'short', evaluate=True, eval_scope='subset')
    assert train.data.val_files.endswith('validation_subset.parquet')
    assert full.data.val_files.endswith('test.parquet')
    assert short.data.val_files.endswith('validation_subset.parquet')
    assert train.ray_init.runtime_env.env_vars.QA_EVAL_MODE == 'auto'
    assert full.ray_init.runtime_env.env_vars.QA_EVAL_MODE == 'full'
    assert short.ray_init.runtime_env.env_vars.QA_EVAL_MODE == 'subset'
    assert train.data.train_files == full.data.train_files == short.data.train_files


def test_report_export_cannot_mislabel_a_monitor_point_as_full_test():
    from report import checked_scope
    sample = manifest()
    value = {'scope': 'subset', 'questions': sample['questions'],
             'ordered_question_ids_sha256': sample['ordered_question_ids_sha256'],
             'by_dataset': {name: {'questions': count} for name, count in sample['source_counts'].items()}}
    assert checked_scope(value) == 'subset'
    with pytest.raises(RuntimeError): checked_scope({**value, 'scope': 'full'})
    with pytest.raises(RuntimeError): checked_scope({**value, 'questions': 447})


def import_wrapper(monkeypatch, native):
    module = ModuleType('verl.trainer.ppo.skillsd_ray_trainer')
    module.SkillSDRayTrainer = native
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.skillsd_ray_trainer', module)
    spec = importlib.util.spec_from_file_location('subset_validation_fixture', ROOT / 'src/grpo_eval.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    return wrapper


def test_loader_switching_and_receipts_keep_scopes_separate_without_touching_training(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    inputs = [dict(question='Q' + str(i), ground_truth={'target': ['A']},
                   data_source='nq' if i < 2 else 'hotpotqa') for i in range(3)]
    full_ids = [question_id(x) for x in inputs]
    subset_ids = [full_ids[i] for i in [0, 2]]
    full_record = {'files': {'test': {'rows': {'count': 3, 'question_ids': full_ids,
                   'source_counts': {'nq': 2, 'hotpotqa': 1}}}}}
    subset_record = {'rows': {'count': 2, 'question_ids': subset_ids,
                     'source_counts': {'nq': 1, 'hotpotqa': 1}}, 'source_test_indices': [0, 2]}
    (tmp_path / 'full.json').write_text(json.dumps(full_record))
    (tmp_path / 'subset.json').write_text(json.dumps(subset_record))
    monkeypatch.setenv('QA_DATA_RECEIPT', str(tmp_path / 'full.json'))
    monkeypatch.setenv('QA_SUBSET_RECEIPT', str(tmp_path / 'subset.json'))
    monkeypatch.setenv('QA_EVAL_MODE', 'auto')
    monkeypatch.setenv('GRPO_RUN_DIR', str(tmp_path / 'run'))

    class Native:
        def _validate(self):
            self.loaders_used.append(self.val_dataloader)
            offset = 0
            for items in self.val_dataloader:
                gen = SimpleNamespace(non_tensor_batch={'env_kwargs': np.array(items, dtype=object),
                    'data_source': np.array([x['data_source'] for x in items])})
                self.traj_collector.multi_turn_loop(gen_batch=gen, is_train=False)
                offset += len(items)
            return {'val/native_unchanged': .25}

    wrapper = import_wrapper(monkeypatch, Native)
    trainer = wrapper.SearchTrainer()
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(val_only=True))
    subset_loader = [[inputs[0], inputs[2]]]
    full_loader = [inputs[:2], inputs[2:]]
    subset_dataset, full_dataset, train_loader = object(), object(), object()
    trainer.val_dataset, trainer.val_dataloader = subset_dataset, subset_loader
    trainer.train_dataloader = train_loader
    trainer._searchqa_evaluation_loaders = {'subset': (subset_dataset, subset_loader), 'full': (full_dataset, full_loader)}
    trainer.loaders_used = []
    trainer.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: '<answer>A</answer>')

    def collect(gen_batch, **kwargs):
        items = gen_batch.non_tensor_batch.pop('env_kwargs')
        local = list(range(len(items)))
        return SimpleNamespace(batch={'responses': torch.ones((len(items), 2), dtype=torch.long),
            'attention_mask': torch.ones((len(items), 4), dtype=torch.long)}, non_tensor_batch={
            'index': local, 'traj_uid': [x['question'] for x in items],
            'episode_rewards': [float(x['question'] != 'Q1') for x in items],
            'data_source': [x['data_source'] for x in items], 'turn_step': [0] * len(items),
            'is_action_valid': [True] * len(items)})

    trainer.traj_collector = SimpleNamespace(multi_turn_loop=collect)
    for step in [0, 20, 140, 150]:
        trainer.global_steps = step
        result = trainer._validate()
        scope = 'full' if step == 150 else 'subset'
        assert result['val/native_unchanged'] == .25
        assert result['eval/' + scope + '/questions'] == (3 if scope == 'full' else 2)
        assert trainer.val_dataset is subset_dataset and trainer.val_dataloader is subset_loader
        assert trainer.train_dataloader is train_loader
        assert trainer.traj_collector.multi_turn_loop is collect
    assert trainer.loaders_used == [subset_loader, subset_loader, subset_loader, full_loader]
    summaries = [json.loads(p.read_text(encoding='utf-8')) for p in (tmp_path / 'run/evaluation').glob('update_*/attempt-*/summary.json')]
    assert len([v for v in summaries if v['scope'] == 'full']) == 1
    assert all(v['questions'] == 2 for v in summaries if v['scope'] == 'subset')
    assert all(v['wall_seconds'] >= 0 and v['monitor_target_is_not_a_timeout'] for v in summaries)


def test_scope_loader_creation_never_recreates_the_training_sampler(tmp_path, monkeypatch):
    wrapper = import_wrapper(monkeypatch, object)
    calls = []
    native_main = ModuleType('verl.trainer.main_ppo')
    native_main.create_rl_dataset = lambda *args: calls.append(args) or ['validation-item']
    monkeypatch.setitem(sys.modules, 'verl.trainer.main_ppo', native_main)
    monkeypatch.setenv('QA_INITIAL_EVAL_SCOPE', 'subset')
    monkeypatch.setenv('QA_DATA_DIRECTORY', str(tmp_path))
    collate = lambda values: values

    class Loader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.collate_fn = kwargs['collate_fn']
        def __len__(self): return 1

    trainer = wrapper.SearchTrainer()
    train_loader, dataset = object(), object()
    trainer.train_dataloader = train_loader
    trainer.val_dataset = dataset
    trainer.val_dataloader = Loader(collate_fn=collate)
    class Data(dict):
        def __getattr__(self, key): return self[key]
    trainer.config = SimpleNamespace(data=Data(val_batch_size=512, dataloader_num_workers=8))
    trainer.tokenizer, trainer.processor = object(), None
    created, loader = trainer._evaluation_loader('full')
    assert len(calls) == 1 and calls[0][0] == str(tmp_path / 'test.parquet')
    assert loader.kwargs['shuffle'] is False and loader.kwargs['drop_last'] is False
    assert loader.kwargs['batch_size'] == 512 and loader.kwargs['collate_fn'] is collate
    assert trainer.train_dataloader is train_loader
    trainer._evaluation_loader('full')
    assert len(calls) == 1
