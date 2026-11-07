"""Small four-rank saves, real CPU torch roundtrip, and failure injection."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
import types
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import checkpoint_io
from checkpoint_io import CheckpointStore, COMPLETE, NativeCheckpointMixin, POINTER
from run import config, run_lock
from common import WORLD_SIZE


def native_files(directory, step, *, torch=None, world_size=WORLD_SIZE):
    directory.mkdir(parents=True)
    actor = directory / 'actor'
    actor.mkdir()
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'):
        (actor / name).write_text('{"fixture": true}', encoding='utf-8')
    (actor / 'merges.txt').write_text('#version: 0.2\na b\n', encoding='utf-8')
    paths = [directory / 'data.pt'] + [actor / f'{kind}_world_size_{world_size}_rank_{rank}.pt'
             for rank in range(world_size) for kind in ('model', 'optim', 'extra_state')]
    for path in paths:
        payload = {'step': step, 'file': path.name}
        if torch is not None:
            payload['tensor'] = torch.tensor([step, 7])
            torch.save(payload, path)
        else:
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('archive/data.pkl', pickle.dumps(payload))
                archive.writestr('archive/version', b'3\n')
                archive.writestr('archive/data/0', b'small tensor storage')


@pytest.fixture
def store(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'binding.json').write_text(json.dumps({'command': 'train', 'world_size': WORLD_SIZE,
                                                 'recipe': 'same-scientific-settings'}), encoding='utf-8')
    return CheckpointStore(run)


def save(store, step, *, torch=None):
    return store.save(step, lambda staging: native_files(staging / f'global_step_{step}', step, torch=torch))


def steps(store):
    return [int(path.name.removeprefix('global_step_')) for path in store.scan()]


def test_save_then_rotate_not_rotate_then_save(store):
    save(store, 20)
    save(store, 40)

    def writer(staging):
        assert steps(store) == [20, 40]
        assert int((store.root / POINTER).read_text(encoding='utf-8')) == 40
        native_files(staging / 'global_step_60', 60)
        (staging / POINTER).write_text('60')  # Native tracker stays private until commit.
        assert steps(store) == [20, 40]

    store.save(60, writer)
    assert steps(store) == [40, 60]
    assert not (store.root / 'global_step_20').exists()
    assert int((store.root / POINTER).read_text(encoding='utf-8')) == 60
    assert not list(store.root.glob('.saving-*'))
    manifest = json.loads((store.root / 'global_step_60' / COMPLETE).read_text(encoding='utf-8'))
    assert manifest['step'] == 60 and manifest['world_size'] == WORLD_SIZE
    assert len(manifest['files']) == 3 * WORLD_SIZE + 6


@pytest.mark.parametrize('failure', ['writer', 'missing_rank', 'missing_data', 'truncated_zip', 'crc', 'json'])
def test_failed_save_never_prunes_or_changes_pointer(store, failure):
    save(store, 20)
    save(store, 40)

    def writer(staging):
        if failure == 'writer':
            raise OSError('simulated full disk / worker failure')
        path = staging / 'global_step_60'
        native_files(path, 60)
        if failure == 'missing_rank':
            (path / f'actor/optim_world_size_{WORLD_SIZE}_rank_{WORLD_SIZE - 1}.pt').unlink()
        elif failure == 'missing_data':
            (path / 'data.pt').unlink()
        elif failure == 'truncated_zip':
            (path / 'data.pt').write_bytes(b'PK\x03\x04incomplete')
        elif failure == 'crc':
            target = path / 'data.pt'
            payload = target.read_bytes().replace(b'small tensor storage', b'fault tensor storage')
            target.write_bytes(payload)
        elif failure == 'json':
            (path / 'actor/tokenizer.json').write_text('{broken')

    with pytest.raises((RuntimeError, OSError, ValueError)):
        store.save(60, writer)
    assert steps(store) == [20, 40]
    assert int((store.root / POINTER).read_text(encoding='utf-8')) == 40
    assert list(store.root.glob('.saving-*'))  # Partial evidence is not silently discarded.


def test_post_seal_sha_failure_does_not_prune(store, monkeypatch):
    save(store, 20)
    save(store, 40)
    certify = store.certify

    def corrupt_after_seal(path, **kwargs):
        certify(path, **kwargs)
        target = path / 'actor/merges.txt'
        target.write_bytes(target.read_bytes().replace(b'a b', b'c d'))  # Same-size corruption.

    monkeypatch.setattr(store, 'certify', corrupt_after_seal)
    with pytest.raises(RuntimeError, match='old checkpoints preserved'):
        save(store, 60)
    assert steps(store) == [20, 40]
    assert int((store.root / POINTER).read_text(encoding='utf-8')) == 40


@pytest.mark.parametrize('pointer', ['missing', 'stale', 'invalid'])
def test_recovery_after_atomic_publish_before_pointer(store, monkeypatch, pointer):
    save(store, 20)
    save(store, 40)

    def crash(step):
        raise OSError('simulated controller crash after publish')

    monkeypatch.setattr(store, '_publish_pointer', crash)
    with pytest.raises(OSError):
        save(store, 60)
    assert steps(store) == [20, 40, 60]
    tracker = store.root / POINTER
    if pointer == 'missing':
        tracker.unlink()
    elif pointer == 'invalid':
        tracker.write_text('interrupted write')
    recovered = CheckpointStore(store.run_dir)
    assert recovered.recover().name == 'global_step_60'
    assert steps(recovered) == [40, 60]
    assert int(tracker.read_text(encoding='utf-8')) == 60


def test_resume_adopts_legacy_disk_saves_and_keeps_two(store):
    for step in (20, 40, 60, 80):
        native_files(store.root / f'global_step_{step}', step)
    (store.root / POINTER).write_text('20')
    # A fresh controller has no in-memory native saved-path list.
    restarted = CheckpointStore(store.run_dir)
    assert restarted.recover().name == 'global_step_80'
    assert steps(restarted) == [60, 80]
    save(restarted, 100)
    assert steps(restarted) == [80, 100]


def test_partial_corrupt_and_foreign_are_not_loaded_or_deleted(store):
    save(store, 20)
    save(store, 40)
    native_files(store.root / 'global_step_50', 50)
    (store.root / 'global_step_50/data.pt').unlink()
    native_files(store.root / 'global_step_60', 60)
    store.certify(store.root / 'global_step_60')
    (store.root / 'global_step_60/actor/merges.txt').write_text('damaged')
    native_files(store.root / 'global_step_80', 80)
    store.certify(store.root / 'global_step_80')
    receipt_path = store.root / 'global_step_80' / COMPLETE
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt['binding_sha256'] = '0' * 64
    receipt_path.write_text(json.dumps(receipt))
    assert store.recover().name == 'global_step_40'
    save(store, 100)
    assert steps(store) == [40, 100]
    for step in (50, 60, 80):
        assert (store.root / f'global_step_{step}').exists()


def test_no_verified_save_is_not_a_fresh_start(store):
    native_files(store.root / 'global_step_20', 20)
    (store.root / 'global_step_20/data.pt').unlink()
    with pytest.raises(RuntimeError, match='No complete checkpoint'):
        store.recover()
    assert not (store.root / POINTER).exists()


def test_wrong_world_size_rejected(store):
    path = store.root / 'global_step_20'
    native_files(path, 20, world_size=8)
    with pytest.raises(RuntimeError, match='Incomplete four-rank'):
        store.certify(path)


def test_eight_rank_run_binding_is_rejected_without_modifying_existing_saves(store):
    save(store, 20)
    checkpoint = store.root / 'global_step_20'
    before = (checkpoint / COMPLETE).read_bytes()
    binding_path = store.run_dir / 'binding.json'
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    binding['world_size'] = 8
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(RuntimeError, match='four-rank training binding'):
        CheckpointStore(store.run_dir)
    assert (checkpoint / COMPLETE).read_bytes() == before
    assert int((store.root / POINTER).read_text(encoding='utf-8')) == 20


def test_exact_inventory_and_manifest_paths_cannot_escape(store, tmp_path):
    path = save(store, 20)
    outside = tmp_path / 'outside.txt'
    outside.write_text('must remain untouched')
    receipt_path = path / COMPLETE
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt['files']['../../outside.txt'] = receipt['files']['data.pt']
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match='inventory differs'):
        store.verify(path)
    with pytest.raises(RuntimeError):
        store.verify(tmp_path)
    assert outside.read_text(encoding='utf-8') == 'must remain untouched'


def test_existing_save_is_not_overwritten(store):
    path = save(store, 20)
    old = (path / COMPLETE).read_bytes()
    with pytest.raises(RuntimeError, match='overwrite'):
        save(store, 20)
    assert (path / COMPLETE).read_bytes() == old


@pytest.mark.skipif(os.name != 'posix', reason='POSIX symlinks and controller lock')
def test_symlink_targets_and_contents_never_deleted(store, tmp_path):
    save(store, 20)
    save(store, 40)
    outside = tmp_path / 'outside'
    native_files(outside / 'global_step_60', 60)
    (store.root / 'global_step_60').symlink_to(outside / 'global_step_60', target_is_directory=True)
    native_files(store.root / 'global_step_80', 80)
    (store.root / 'global_step_80/actor/merges.txt').unlink()
    (store.root / 'global_step_80/actor/merges.txt').symlink_to(outside / 'global_step_60/actor/merges.txt')
    store.recover()
    save(store, 100)
    assert steps(store) == [40, 100]
    assert (outside / 'global_step_60/data.pt').exists()
    assert (store.root / 'global_step_60').is_symlink()
    assert (store.root / 'global_step_80/actor/merges.txt').is_symlink()
    with pytest.raises(RuntimeError, match='running controller'):
        with run_lock(store.run_dir), run_lock(store.run_dir):
            pass


def test_interrupted_pruning_leaves_two_healthy_newest(store, monkeypatch):
    save(store, 20)
    save(store, 40)
    remove = checkpoint_io.shutil.rmtree

    def partial_remove(path):
        (path / 'data.pt').unlink()
        raise OSError('interrupted during old checkpoint cleanup')

    monkeypatch.setattr(checkpoint_io.shutil, 'rmtree', partial_remove)
    with pytest.raises(OSError):
        save(store, 60)
    assert steps(store) == [40, 60]
    assert int((store.root / POINTER).read_text(encoding='utf-8')) == 60
    monkeypatch.setattr(checkpoint_io.shutil, 'rmtree', remove)
    assert CheckpointStore(store.run_dir).recover().name == 'global_step_60'
    assert steps(store) == [40, 60]


def adapter(store, *, fail=False):
    class Native:
        def _save_checkpoint(self):
            trainer = self.config.trainer
            assert trainer.max_actor_ckpt_to_keep is None
            assert trainer.max_critic_ckpt_to_keep is None
            assert Path(trainer.default_local_dir).name.startswith('.saving-')
            assert (store.root / f'global_step_{self.global_steps}').exists() is False
            if fail:
                raise OSError('native serialization failure')
            native_files(Path(trainer.default_local_dir) / f'global_step_{self.global_steps}', self.global_steps)

        def _load_checkpoint(self):
            self.loaded_from = self.config.trainer.resume_from_path
            return 'native loader'

    class Trainer(NativeCheckpointMixin, Native):
        pass

    trainer = Trainer()
    trainer.config = config('1.5b', store.run_dir / 'model', store.run_dir)
    trainer.global_steps = 20
    trainer.use_critic = False
    return trainer


@pytest.mark.parametrize('fail', [False, True])
def test_adapter_only_changes_storage_and_restores_config_even_on_failure(store, monkeypatch, fail):
    from omegaconf import OmegaConf
    monkeypatch.setenv('GRPO_RUN_DIR', str(store.run_dir))
    trainer = adapter(store, fail=fail)
    before = OmegaConf.to_container(trainer.config, resolve=True)
    rng = random.getstate()
    if fail:
        with pytest.raises(OSError):
            trainer._save_checkpoint()
    else:
        trainer._save_checkpoint()
        assert steps(store) == [20]
    assert OmegaConf.to_container(trainer.config, resolve=True) == before
    assert random.getstate() == rng


def test_adapter_loads_verified_recovered_exact_path(store, monkeypatch):
    monkeypatch.setenv('GRPO_RUN_DIR', str(store.run_dir))
    save(store, 20)
    save(store, 40)
    trainer = adapter(store)
    with pytest.raises(RuntimeError, match='explicit --resume'):
        trainer._load_checkpoint()
    checkpoint = CheckpointStore(store.run_dir).recover()
    trainer.config.trainer.resume_mode = 'resume_path'
    trainer.config.trainer.resume_from_path = str(checkpoint)
    assert trainer._load_checkpoint() == 'native loader'
    assert Path(trainer.loaded_from) == checkpoint
    (checkpoint / 'data.pt').write_bytes(b'corruption')
    with pytest.raises(RuntimeError, match='mismatch'):
        trainer._load_checkpoint()


def test_independent_hf_evaluation_does_not_require_training_checkpoints(store, monkeypatch):
    monkeypatch.setenv('GRPO_RUN_DIR', str(store.run_dir))
    trainer = adapter(store)
    binding_path = store.run_dir / 'binding.json'
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    binding['command'] = 'eval'
    binding_path.write_text(json.dumps(binding))
    trainer.config.trainer.val_only = True
    assert trainer._load_checkpoint() == 'native loader'
    assert trainer.loaded_from is None


def test_real_cpu_torch_save_restore_and_retention(store):
    torch = pytest.importorskip('torch')
    rng = torch.get_rng_state().clone()
    for step in (20, 40, 60):
        save(store, step, torch=torch)
    assert torch.equal(torch.get_rng_state(), rng)
    restored = CheckpointStore(store.run_dir).recover()
    assert steps(store) == [40, 60]
    for name in json.loads((restored / COMPLETE).read_text(encoding='utf-8'))['files']:
        if name.endswith('.pt'):
            payload = torch.load(restored / name, weights_only=True, map_location='cpu')
            assert payload['step'] == 60
            assert payload['tensor'].tolist() == [60, 7]


def test_mixin_calls_actual_locked_native_save_and_load(store, monkeypatch):
    """Execute the original controller methods, with tiny CPU worker stand-ins."""
    torch = pytest.importorskip('torch')
    source = ROOT / '.deps/sdar/verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    native = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == 'RayPPOTrainer')
    methods = [node for node in native.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_save_checkpoint', '_load_checkpoint')]
    namespace = {'os': os, 'torch': torch, 'find_latest_ckpt_path': lambda root:
                 str(Path(root) / ('global_step_' + (Path(root) / POINTER).read_text(encoding='utf-8').strip()))}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    Native = type('LockedNative', (), {name: namespace[name]
                  for name in ('_save_checkpoint', '_load_checkpoint')})

    class Trainer(NativeCheckpointMixin, Native):
        pass

    monkeypatch.setenv('GRPO_RUN_DIR', str(store.run_dir))
    trainer = Trainer()
    trainer.config = config('1.5b', store.run_dir / 'model', store.run_dir)
    trainer.use_critic = False
    loaded = {}

    class WorkerGroup:
        def save_checkpoint(self, local, remote, step, max_ckpt_to_keep):
            assert remote is None and max_ckpt_to_keep is None
            if step == 60:
                assert steps(store) == [20, 40]
            native_files(Path(local).parent, step, torch=torch)

        def load_checkpoint(self, local, del_local_after_load):
            assert not del_local_after_load
            for rank in range(WORLD_SIZE):
                for kind in ('model', 'optim', 'extra_state'):
                    file = Path(local) / f'{kind}_world_size_{WORLD_SIZE}_rank_{rank}.pt'
                    value = torch.load(file, weights_only=True, map_location='cpu')
                    assert value['tensor'].tolist() == [60, 7]
            loaded['worker_step'] = 60

    trainer.actor_rollout_wg = WorkerGroup()
    trainer.train_dataloader = types.SimpleNamespace(
        state_dict=lambda: {'cursor': trainer.global_steps},
        load_state_dict=lambda state: loaded.update(dataloader=state))
    for step in (20, 40, 60):
        trainer.global_steps = step
        trainer._save_checkpoint()
    assert steps(store) == [40, 60]
    trainer.config.trainer.resume_mode = 'resume_path'
    trainer.config.trainer.resume_from_path = str(store.recover())
    trainer.global_steps = 0
    trainer._load_checkpoint()
    assert trainer.global_steps == 60
    assert loaded == {'worker_step': 60, 'dataloader': {'cursor': 60}}


def test_trainer_mro_keeps_native_fit_save_and_load_dispatch(monkeypatch):
    class Native:
        def fit(self):
            self._load_checkpoint()
            self._save_checkpoint()
            return 'native fit'

    module = types.ModuleType('verl.trainer.ppo.skillsd_ray_trainer')
    module.SkillSDRayTrainer = Native
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.skillsd_ray_trainer', module)
    spec = importlib.util.spec_from_file_location('storage_grpo_eval', ROOT / 'src/grpo_eval.py')
    evaluated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluated)
    assert evaluated.SearchTrainer.__mro__[1:3] == (NativeCheckpointMixin, Native)
    assert evaluated.SearchTrainer._save_checkpoint is NativeCheckpointMixin._save_checkpoint
    assert evaluated.SearchTrainer._load_checkpoint is NativeCheckpointMixin._load_checkpoint


def test_storage_adapter_retains_original_native_algorithm():
    from setup import verify
    lock = verify()
    for name in ('verl/trainer/ppo/core_algos.py', 'verl/workers/actor/dp_actor.py',
                 'verl/workers/fsdp_workers.py', 'verl/trainer/ppo/ray_trainer.py'):
        assert name in lock['unchanged_files']
    assert set(lock['patched_files']) == {'verl/trainer/main_sdar.py',
                                          'agent_system/multi_turn_rollout/rollout_loop.py'}


def search_wrapper(monkeypatch, native):
    module = types.ModuleType('verl.trainer.ppo.skillsd_ray_trainer')
    module.SkillSDRayTrainer = native
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.skillsd_ray_trainer', module)
    spec = importlib.util.spec_from_file_location('final_checkpoint_search_eval', ROOT / 'src/grpo_eval.py')
    wrapped = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapped)
    return wrapped


@pytest.mark.parametrize('evaluation_fails', [False, True])
def test_final_checkpoint_is_verified_before_full_test_and_never_written_twice(store, monkeypatch, evaluation_fails):
    save(store, 120)
    save(store, 140)
    events = []

    class Native:
        def _save_checkpoint(self):
            events.append('save')
            assert self.config.trainer.max_actor_ckpt_to_keep is None
            native_files(Path(self.config.trainer.default_local_dir) / 'global_step_150', 150)

    wrapper = search_wrapper(monkeypatch, Native)
    trainer = wrapper.SearchTrainer()
    trainer.config = config('3b', store.run_dir / 'model', store.run_dir)
    trainer.use_critic, trainer.global_steps = False, 150
    dataset, loader = object(), object()
    trainer.val_dataset, trainer.val_dataloader = dataset, loader
    trainer._evaluation_loader = lambda scope: (object(), object())
    monkeypatch.setenv('GRPO_RUN_DIR', str(store.run_dir))
    monkeypatch.setenv('QA_EVAL_MODE', 'auto')

    def full_test(scope, started):
        events.append('evaluate')
        assert scope == 'full'
        store.verify(store.root / 'global_step_150')
        assert steps(store) == [140, 150]
        if evaluation_fails:
            raise TimeoutError('interrupted full evaluation')
        return {'eval/full/question_micro_em': .5}

    trainer._validate_selected = full_test
    if evaluation_fails:
        with pytest.raises(TimeoutError, match='interrupted full evaluation'):
            trainer._validate()
    else:
        assert trainer._validate()['eval/full/question_micro_em'] == .5
    assert events == ['save', 'evaluate']
    assert trainer.val_dataset is dataset and trainer.val_dataloader is loader
    store.verify(store.root / 'global_step_150')
    # This is the subsequent call made by the untouched native fit loop.
    trainer._save_checkpoint()
    assert events == ['save', 'evaluate']
    assert steps(store) == [140, 150]
    target = store.root / 'global_step_150/actor/merges.txt'
    target.write_bytes(target.read_bytes().replace(b'a b', b'c d'))
    with pytest.raises(RuntimeError, match='mismatch'):
        trainer._save_checkpoint()


def test_failed_final_save_does_not_start_expensive_test_or_remove_old_checkpoints(store, monkeypatch):
    save(store, 120)
    save(store, 140)

    class Native:
        def _save_checkpoint(self):
            raise OSError('final serializer interrupted')

    wrapper = search_wrapper(monkeypatch, Native)
    trainer = wrapper.SearchTrainer()
    trainer.config = config('3b', store.run_dir / 'model', store.run_dir)
    trainer.use_critic, trainer.global_steps = False, 150
    trainer._evaluation_loader = lambda scope: pytest.fail('evaluation must not start before a verified save')
    monkeypatch.setenv('GRPO_RUN_DIR', str(store.run_dir))
    monkeypatch.setenv('QA_EVAL_MODE', 'auto')
    with pytest.raises(OSError, match='serializer interrupted'):
        trainer._validate()
    assert steps(store) == [120, 140]
    assert int((store.root / POINTER).read_text(encoding='utf-8')) == 140


@pytest.mark.parametrize('step', [0, 150])
def test_independent_full_evaluation_does_not_save_a_training_checkpoint(tmp_path, monkeypatch, step):
    wrapper = search_wrapper(monkeypatch, object)
    trainer = wrapper.SearchTrainer()
    trainer.config = config('3b', tmp_path / 'model', tmp_path / 'eval', evaluate=True)
    trainer.global_steps = step
    trainer.val_dataset, trainer.val_dataloader = object(), object()
    trainer._evaluation_loader = lambda scope: (object(), object())
    trainer._validate_selected = lambda scope, started: {'scope': scope}
    trainer._save_checkpoint = lambda: pytest.fail('evaluation-only must not save training state')
    monkeypatch.setenv('QA_EVAL_MODE', 'full')
    assert trainer._validate() == {'scope': 'full'}
    assert not (tmp_path / 'eval').exists()


@pytest.mark.parametrize('step', [20, 140, 150])
def test_resume_at_final_update_only_retries_evaluation(tmp_path, step):
    from run import configure_resume
    cfg = config('3b', tmp_path / 'model', tmp_path / 'run')
    path = tmp_path / 'run/checkpoints' / f'global_step_{step}'
    configure_resume(cfg, path)
    assert cfg.trainer.resume_mode == 'resume_path'
    assert cfg.trainer.resume_from_path == str(path)
    assert cfg.trainer.val_only == (step == 150)
    assert cfg.trainer.total_training_steps == 150


@pytest.mark.parametrize('step', [0, 151])
def test_resume_rejects_zero_and_beyond_training_budget(tmp_path, step):
    from run import configure_resume
    cfg = config('3b', tmp_path / 'model', tmp_path / 'run')
    with pytest.raises(RuntimeError, match='complete checkpoint'):
        configure_resume(cfg, tmp_path / f'global_step_{step}')


def test_final_resume_exits_actual_native_fit_before_update151(store, monkeypatch):
    from run import configure_resume
    source = ROOT / '.deps/sdar/verl/trainer/ppo/skillsd_ray_trainer.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SkillSDRayTrainer')
    method = next(n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    namespace = {'pprint': lambda *args, **kwargs: None,
                 'tqdm': lambda **kwargs: pytest.fail('completed final update must not restart training')}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)

    class Native:
        fit = namespace['fit']

        def _load_checkpoint(self):
            self.global_steps = 150

    class Tracking:
        def __init__(self, **kwargs):
            pass
        def log(self, **kwargs):
            pass

    tracking = types.ModuleType('verl.utils.tracking')
    tracking.Tracking = Tracking
    monkeypatch.setitem(sys.modules, 'verl.utils.tracking', tracking)
    wrapper = search_wrapper(monkeypatch, Native)
    trainer = wrapper.SearchTrainer()
    trainer.config = config('3b', store.run_dir / 'model', store.run_dir)
    configure_resume(trainer.config, store.root / 'global_step_150')
    trainer.val_reward_fn = object()
    calls = []
    trainer._validate = lambda: calls.append(trainer.global_steps) or {'eval/full/questions': 51713}
    monkeypatch.setenv('GRPO_RUN_DIR', str(store.run_dir))
    trainer.fit()
    assert calls == [150] and trainer.global_steps == 150
