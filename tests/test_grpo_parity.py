"""Lock the exact shared GRPO benchmark while allowing only named method changes."""
import ast
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import load, sha
from run import assert_sdar_only, config


def test_shared_files_match_both_public_grpo_references_byte_for_byte():
    lock = load('grpo_comparison.json')
    assert set(lock['reference_repositories']) == {
        'leiye302/grpo-searchqa-qwen25-1.5b', 'leiye302/grpo-searchqa-qwen25-3b'}
    for name, expected in lock['shared_file_sha256'].items():
        assert sha(ROOT / name) == expected, name


def test_evaluation_methods_are_exact_same_source_as_published_grpo():
    source = (ROOT / 'src/grpo_eval.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SearchTrainer')
    # Python3.12 adds FunctionDef.type_params; hash original method source
    # instead of a Python-version-specific AST dump. No assertion is weakened.
    bodies = [ast.get_source_segment(source, n) for n in cls.body]
    identity = hashlib.sha256('\n'.join(bodies).encode()).hexdigest()
    assert identity == load('grpo_comparison.json')['evaluation_class_methods_source_sha256']
    assert ast.unparse(cls.bases[-1]) == 'SkillSDRayTrainer'


def test_only_named_method_changes_to_grpo_configuration():
    lock = load('grpo_comparison.json')
    reference = dict(lock['reference_experiment'])
    reference.update(lock['named_changes'])
    assert load('experiment.json') == reference
    assert load('evaluation.json')['monitor_subset']['questions'] == 448
    assert load('evaluation.json')['full_validation_updates'] == [150]


@pytest.mark.parametrize('key,value', [
    ('use_sdar_loss', False), ('sdar_gate_beta', 0.), ('sdar_loss_coef', .1),
])
def test_conflicting_actor_side_overrides_are_rejected(tmp_path, key, value):
    from omegaconf import OmegaConf
    cfg = config('3b', tmp_path / 'model', tmp_path / 'run')
    OmegaConf.update(cfg, 'actor_rollout_ref.actor.' + key, value, force_add=True)
    with pytest.raises(RuntimeError, match='disable|Conflicting'):
        assert_sdar_only(cfg)


def test_native_fit_keeps_teacher_and_unchanged_advantages():
    source = (ROOT / '.deps/sdar/verl/trainer/ppo/skillsd_ray_trainer.py').read_text(encoding='utf-8')
    assert 'teacher_log_probs, teacher_metrics = self._get_training_teacher_log_probs(batch)' in source
    assert 'batch = compute_advantage(' in source
    assert 'batch.meta_info["sdl_coef"] = current_sdl_lambda' in source
    assert 'batch = self._apply_token_level_advantages(' not in source
    trainer = ast.parse(source)
    cls = next(n for n in trainer.body if isinstance(n, ast.ClassDef) and n.name == 'SkillSDRayTrainer')
    assert not any(isinstance(n, ast.FunctionDef) and n.name == '_validate' for n in cls.body)
