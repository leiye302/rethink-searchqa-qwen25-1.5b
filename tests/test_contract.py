import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import UPSTREAM, WORLD_SIZE, devices, load
from evaluation_contract import assert_protocol
from run import assert_sdar_only, config
from setup import verify


def test_native_sources_and_only_teacher_entry_and_rollout_metadata_patch():
    lock = verify()
    assert len(lock['unchanged_files']) >= 20
    assert lock['native_launcher'] == 'examples/grpo_opsd_trainer/run_search_3b.sh'
    assert set(lock['patched_files']) == {'verl/trainer/main_sdar.py',
                                          'agent_system/multi_turn_rollout/rollout_loop.py'}
    source = (UPSTREAM / 'verl/trainer/main_sdar.py').read_text(encoding='utf-8')
    assert 'from rethink_searchqa.trainer import RethinkSearchTrainer as SkillSDRayTrainer' in source
    patch=(ROOT/'patches/runtime.patch').read_text(encoding='utf-8')
    additions=[x[1:].strip() for x in patch.splitlines() if x.startswith('+') and not x.startswith('+++')]
    assert len(additions)==8
    assert all(any(key in x for key in ('rethink_records','rethink_searchqa')) for x in additions)
    assert 'verl/trainer/ppo/ray_trainer.py' in lock['unchanged_files']
    assert 'agent_system/environments/env_manager.py' in lock['unchanged_files']


def test_only_public_fetch_has_bounded_transfer_retries(tmp_path, monkeypatch):
    import setup
    calls, waits = [], []
    def transfer(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0 if len(calls) == 3 else 1,
                                           stdout='pinned source', stderr='transfer interrupted')
    monkeypatch.setattr(setup.subprocess, 'run', transfer)
    monkeypatch.setattr(setup.time, 'sleep', waits.append)
    assert setup.git(tmp_path, 'fetch', 'origin', 'fixed-revision') == 'pinned source'
    assert len(calls) == 3 and waits == [1, 2]
    with pytest.raises(subprocess.CalledProcessError):
        setup.git(tmp_path, 'checkout', 'fixed-revision')
    assert len(calls) == 4 and waits == [1, 2]


@pytest.mark.parametrize('size', ['1.5b', '3b'])
def test_native_composed_config(size, tmp_path):
    cfg = config(size, tmp_path / 'model', tmp_path / 'run')
    assert cfg.trainer.n_gpus_per_node == cfg.ray_init.num_gpus == WORLD_SIZE == 4
    assert cfg.data.train_batch_size == 128 and cfg.env.rollout.n == 8
    assert cfg.actor_rollout_ref.rollout.n == 1
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 256
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 16
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio == .1
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == .001
    assert cfg.actor_rollout_ref.actor.invalid_action_penalty_coef == .01
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == .5
    assert cfg.algorithm.sdar.gate_beta == 5.
    assert cfg.algorithm.sdar.sdar_coef == .01
    assert cfg.algorithm.sdar.skills_dir == str(UPSTREAM / 'skills/search')
    assert cfg.trainer.total_training_steps == 150 and cfg.trainer.test_freq == 20
    assert str(cfg.data.val_files).endswith('validation_subset.parquet')
    assert cfg.trainer.save_freq == 20 and cfg.trainer.max_actor_ckpt_to_keep == 2
    assert_protocol(cfg)


def test_scientific_overrides_match_original_launcher(tmp_path):
    from omegaconf import OmegaConf
    source = (UPSTREAM / load('upstream.json')['native_launcher']).read_text(encoding='utf-8')
    variables = dict(re.findall(r'(?m)^([A-Za-z_]\w*)=([^\s\n]+)$', source))
    cfg = config('3b', tmp_path/'model', tmp_path/'run')
    # These are precisely the portable/output/storage changes, not loss changes.
    permitted = {'data.train_files', 'data.val_files', 'actor_rollout_ref.model.path',
        'env.search.search_url', 'trainer.logger', 'trainer.project_name', 'trainer.experiment_name',
        'trainer.save_freq', 'trainer.test_freq', 'trainer.val_before_train'}
    for key, raw in re.findall(r'(?m)^[ \t]+([\w.]+)=([^\s\\]+)', source):
        if '.' not in key or key in permitted:
            continue
        if raw.startswith('$'):
            raw = 'vllm' if raw == '$ENGINE' else variables[raw[1:]]
        raw = raw.strip('\x27\x22')
        try:
            expected = json.loads(raw.lower() if raw.lower() in ('true', 'false') else raw)
        except json.JSONDecodeError:
            expected = raw
        assert OmegaConf.select(cfg, key) == expected, key


@pytest.mark.parametrize('name', ['skillsd', 'rlsd', 'opd', 'opsd', 'local_hints'])
def test_no_other_method_can_sneak_into_grpo_opsd(name, tmp_path):
    from omegaconf import OmegaConf
    cfg = config('3b', tmp_path/'model', tmp_path/'run')
    OmegaConf.update(cfg, 'algorithm.' + name, {}, force_add=True)
    with pytest.raises(RuntimeError, match='Unexpected auxiliary'):
        assert_sdar_only(cfg)


@pytest.mark.parametrize('name', ['use_sdl_loss', 'use_opd_loss'])
def test_no_auxiliary_actor_distillation(name, tmp_path):
    from omegaconf import OmegaConf
    cfg = config('3b', tmp_path/'model', tmp_path/'run')
    OmegaConf.update(cfg, 'actor_rollout_ref.actor.' + name, True, force_add=True)
    with pytest.raises(RuntimeError, match='auxiliary'):
        assert_sdar_only(cfg)


def test_original_grpo_loss_has_gradient_and_parameter_update():
    torch = pytest.importorskip('torch')
    from collections import defaultdict
    source = UPSTREAM / 'verl/trainer/ppo/core_algos.py'
    names = {'compute_grpo_outcome_advantage', 'agg_loss', 'compute_policy_loss'}
    functions = [n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'torch': torch, 'np': np, 'defaultdict': defaultdict,
        'verl_F': SimpleNamespace(masked_mean=lambda x,m:(x*m).sum()/m.sum())}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), scope)
    mask = torch.tensor([[1.,1.,0.]] * 4)
    rewards = torch.tensor([[0.,1.,0.],[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]])
    advantages, _ = scope['compute_grpo_outcome_advantage'](rewards, mask,
        np.array(['a','a','b','b']), np.array(['a0','a1','b0','b1']))
    student = torch.nn.Parameter(torch.full_like(mask, -1.))
    old = student.detach().clone()
    optimizer = torch.optim.SGD([student], lr=.1)
    loss = scope['compute_policy_loss'](old, student, advantages, mask, cliprange=.2)[0]
    loss.backward()
    assert student.grad[0,0] < 0 and student.grad[1,0] > 0
    assert torch.count_nonzero(student.grad[:,2]) == 0
    optimizer.step()
    assert student[0,0] > old[0,0] and student[1,0] < old[1,0]


def test_native_validation_is_complete_ordered_and_greedy():
    source = (UPSTREAM/'verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
    loader = ast.unparse(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_create_dataloader'))
    assert 'shuffle=False' in loader and 'drop_last=False' in loader
    evaluation = ast.unparse(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_validate'))
    assert 'for test_data in self.val_dataloader' in evaluation
    assert 'break' not in evaluation
    assert load('data.json')['splits']['test']['rows'] == 51713
    assert sum(load('data.json')['splits']['test']['source_counts'].values()) == 51713


def test_single_backbone_entrypoints():
    size = load('release.json')['size']
    assert size in ('1.5b', '3b')
    for name in ('train.sh', 'eval.sh'):
        assert '--size ' + size in (ROOT/'scripts'/name).read_text(encoding='utf-8')


@pytest.mark.parametrize('spec', ['0,1', '0,1,2,2', '0,1,2,3,4', ''])
def test_gpu_binding_is_four_allocated_cards(spec):
    with pytest.raises(ValueError):
        devices(spec)
    assert devices('4,5,6,7') == ['4','5','6','7']


def test_dry_run_requires_no_weights_data_retriever_or_cuda(tmp_path):
    size = load('release.json')['size']
    result = subprocess.run([sys.executable, str(ROOT/'src/run.py'), 'train', '--size', size,
        '--run-dir', str(tmp_path/'unused'), '--dry-run'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path/'unused').exists()
    assert 'val_batch_size: 512' in result.stdout and 'do_sample: false' in result.stdout
    assert 'env_name: search' in result.stdout


def test_release_privacy_allowlist_and_history():
    result = subprocess.run([sys.executable, str(ROOT/'scripts/check_release.py')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
