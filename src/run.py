"""Search-QA RETHINK Stage 2 on the unchanged four-GPU beta5 Search-QA recipe."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
from common import ROOT, UPSTREAM, WORLD_SIZE, atomic_json, devices, load, sha
from checkpoint_io import CheckpointStore
from evaluation_contract import assert_protocol
from setup import verify
from rethink_searchqa.stage1_assets import expected_model, verify as verify_sft_actor


def base_model(size):
    return ROOT / 'models' / load('models.json')[size]['model_id'].split('/')[-1]


def configure_resume(cfg, checkpoint):
    """Resume updates normally, or finish evaluation of an already-saved final update."""
    step = int(checkpoint.name.removeprefix('global_step_'))
    if not 0 < step <= cfg.trainer.total_training_steps:
        raise RuntimeError('Expected a complete checkpoint at or below the final update')
    cfg.trainer.resume_mode = 'resume_path'
    cfg.trainer.resume_from_path = str(checkpoint)
    # Native fit loads first, validates, then returns when val_only is true.
    # It must not execute update151 after an interrupted final full test.
    cfg.trainer.val_only = step == cfg.trainer.total_training_steps


def assert_sdar_only(cfg):
    if cfg.algorithm.adv_estimator != 'grpo':
        raise RuntimeError('GRPO+OPSD retains native GRPO advantages')
    if ('sdar' not in cfg.algorithm or cfg.algorithm.sdar.sdar_coef != .01
            or cfg.algorithm.sdar.gate_beta != 5. or cfg.algorithm.sdar.skill_all is not False
            or cfg.algorithm.sdar.skills_dir != str(UPSTREAM / 'skills/search')):
        raise RuntimeError('Expected native Search-QA GRPO+OPSD coefficient, beta5 and static skills')
    for name in ('skillsd', 'rlsd', 'opsd', 'opd', 'local_hints'):
        if name in cfg.algorithm:
            raise RuntimeError('Unexpected auxiliary distillation configuration: ' + name)
    for name in ('use_sdl_loss', 'use_opd_loss'):
        if cfg.actor_rollout_ref.actor.get(name, False):
            raise RuntimeError('Unexpected auxiliary Actor loss: ' + name)
    if cfg.get('opd_teacher') is not None or cfg.actor_rollout_ref.actor.get('pg_loss_coef', 1.) != 1.:
        raise RuntimeError('Expected current-Actor Teacher and unchanged GRPO policy loss')
    actor = cfg.actor_rollout_ref.actor
    if actor.get('use_sdar_loss') is False:
        raise RuntimeError('Do not disable the native SDAR loss')
    for name, expected in (('sdar_gate_beta', 5.), ('sdar_loss_coef', .01)):
        if name in actor and actor[name] != expected:
            raise RuntimeError('Conflicting native Actor SDAR setting: ' + name)
    assert_protocol(cfg)


def config(size, model, run_dir, cpus=32, evaluate=False, data_dir=None, search_url=None, eval_scope='full'):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    data_dir = Path(data_dir or ROOT / 'data/searchqa')
    with initialize_config_dir(config_dir=str(UPSTREAM / 'verl/trainer/config'), version_base=None):
        cfg = compose(config_name='ppo_trainer')
    for key, value in load('experiment.json').items():
        OmegaConf.update(cfg, key, value, force_add=True)
    if eval_scope not in {'full', 'subset'}:
        raise ValueError('Independent evaluation scope must be full or subset')
    initial_scope = eval_scope if evaluate else 'subset'
    updates = {
        'actor_rollout_ref.model.path': str(model),
        'algorithm.sdar.skills_dir': str(UPSTREAM / 'skills/search'),
        'data.train_files': str(data_dir / 'train.parquet'),
        'data.val_files': str(data_dir / ('test.parquet' if initial_scope == 'full' else 'validation_subset.parquet')),
        'trainer.experiment_name': 'rethink_searchqa_qwen25_' + size,
        'trainer.default_local_dir': str(run_dir / 'checkpoints'),
        'trainer.val_only': evaluate,
        'trainer.validation_data_dir': str(run_dir / 'validation_native'),
        'ray_init.address': 'local', 'ray_init.num_gpus': WORLD_SIZE,
        'ray_init.num_cpus': cpus, 'ray_init.include_dashboard': False,
        'ray_init._temp_dir': None, 'ray_init.object_store_memory': 2 * 1024**3,
        'ray_init.runtime_env.env_vars.PYTHONPATH': os.pathsep.join([str(ROOT / 'src'), str(UPSTREAM)]),
        'ray_init.runtime_env.env_vars.GRPO_RUN_DIR': str(run_dir),
        'ray_init.runtime_env.env_vars.QA_DATA_RECEIPT': str(data_dir / 'data_receipt.json'),
        'ray_init.runtime_env.env_vars.QA_DATA_DIRECTORY': str(data_dir),
        'ray_init.runtime_env.env_vars.QA_SUBSET_RECEIPT': str(data_dir / 'validation_subset_receipt.json'),
        'ray_init.runtime_env.env_vars.QA_INITIAL_EVAL_SCOPE': initial_scope,
        'ray_init.runtime_env.env_vars.QA_EVAL_MODE': eval_scope if evaluate else 'auto',
        'ray_init.runtime_env.env_vars.TOKENIZERS_PARALLELISM': 'false',
        'ray_init.runtime_env.env_vars.VLLM_USE_V1': '1',
        'ray_init.runtime_env.env_vars.OPENBLAS_NUM_THREADS': '1',
        'ray_init.runtime_env.env_vars.MKL_NUM_THREADS': '1',
        'ray_init.runtime_env.env_vars.OMP_NUM_THREADS': '1',
    }
    if search_url is not None:
        updates['env.search.search_url'] = search_url
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    OmegaConf.resolve(cfg)
    assert_sdar_only(cfg)
    return cfg


@contextlib.contextmanager
def run_lock(run_dir):
    import fcntl
    with (run_dir / 'run.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This output directory already has a running controller') from None
        yield


def check_retriever(url):
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or parsed.username or parsed.password:
        raise ValueError('Use a credential-free retrieval URL; do not put secrets into training configs')
    body = json.dumps({'query': 'What is the capital of France?', 'topk': 3, 'return_scores': True}).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.load(response)
    if len(result.get('result', [])) != 1 or len(result['result'][0]) != 3:
        raise RuntimeError('Retriever must return the native one-query top-3 response schema')
    for row in result['result'][0]:
        if not isinstance(row.get('document'), dict) or not isinstance(row['document'].get('contents'), str):
            raise RuntimeError('Retriever document schema differs from native Search-R1')


@contextlib.contextmanager
def hint_credentials(cfg, enabled, runtime, allow_cost):
    if not enabled:
        yield
        return
    if not allow_cost:
        raise ValueError('Use --allow-api-cost for training with Actor-triggered API hints')
    from rethink_searchqa.hint_api import MODEL, validate_auth, trust_environment_proxy
    value = validate_auth({'base':os.environ.get('HINT_API_BASE'),
                           'model':os.environ.get('HINT_MODEL',MODEL),
                           'key':os.environ.pop('HINT_API_KEY',None)})
    path = runtime / 'hint_auth.json'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream)
        del value
        from omegaconf import OmegaConf
        OmegaConf.update(cfg,'ray_init.runtime_env.env_vars.RETHINK_HINT_AUTH',str(path),force_add=True)
        OmegaConf.update(cfg,'ray_init.runtime_env.env_vars.RETHINK_HINT_TRUST_ENV',
                         'true' if trust_environment_proxy() else 'false',force_add=True)
        yield
    finally:
        path.unlink(missing_ok=True)


def launch(args):
    verify()
    gpu_ids = devices(args.gpus)
    model = (args.model or expected_model()).resolve()
    run_dir = (args.run_dir or ROOT / 'runs' / ('rethink_' + args.size)).resolve()
    data_dir = args.data_dir.resolve()
    cfg = config(args.size, model, run_dir, args.cpus, args.command == 'eval', data_dir, args.search_url,
                 getattr(args, 'eval_scope', None) or 'full')
    if args.dry_run:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg))
        return
    if sys.platform != 'linux':
        raise RuntimeError('Native CUDA GRPO+OPSD requires the documented Linux environment')
    from data import verify as verify_data
    from retrieval import verify as verify_retrieval
    data_receipt = verify_data(data_dir)
    retrieval_receipt = verify_retrieval(args.retrieval_dir.resolve())
    actor_identity = verify_sft_actor(model, args.size, evaluation=args.command == 'eval')
    check_retriever(cfg.env.search.search_url)
    environment = dict(cfg.ray_init.runtime_env.env_vars)
    os.environ.update(environment, CUDA_VISIBLE_DEVICES=','.join(gpu_ids), RAY_ADDRESS='local')
    binding = {'size': args.size, 'model': str(model), 'data_directory': str(data_dir),
        'recipe_sha256': sha(ROOT / 'configs/experiment.json'),
        'upstream_lock_sha256': sha(ROOT / 'configs/upstream.json'),
        'data_lock_sha256': sha(ROOT / 'configs/data.json'),
        'data_receipt_sha256': sha(data_dir / 'data_receipt.json'),
        'evaluation_protocol_sha256': sha(ROOT / 'configs/evaluation.json'),
        'subset_manifest_sha256': sha(ROOT / 'configs/validation_subset.json'),
        'subset_receipt_sha256': sha(data_dir / 'validation_subset_receipt.json'),
        'retrieval_receipt': retrieval_receipt, 'search_url': cfg.env.search.search_url,
        'test_question_ids_sha256': data_receipt['files']['test']['rows']['ordered_question_ids_sha256'],
        'world_size': WORLD_SIZE, 'command': args.command, 'actor_identity': actor_identity,
        'hint_model': 'glm-5.3-flash' if args.command=='train' else None,
        'release_code_sha256': {p.relative_to(ROOT).as_posix(): sha(p)
                                for p in sorted((ROOT / 'src').rglob('*.py'))}}
    if args.resume:
        if args.command != 'train' or json.loads((run_dir / 'binding.json').read_text(encoding='utf-8')) != binding:
            raise RuntimeError('Resume identity differs from the original run')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir / 'binding.json', binding)
    with run_lock(run_dir):
        if args.resume:
            checkpoint = CheckpointStore(run_dir).recover()
            configure_resume(cfg, checkpoint)
        import torch
        if torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError('Four allocated CUDA devices are required')
        for i in range(WORLD_SIZE):
            if torch.cuda.get_device_properties(i).total_memory < 70 * 1024**3:
                raise RuntimeError('This unmodified microbatch recipe targets four A100 80GB GPUs')
        runtime = Path(tempfile.mkdtemp(prefix='so-', dir=os.environ.get('OPSD_RUNTIME_ROOT')))
        cfg.ray_init._temp_dir = str(runtime)
        from omegaconf import OmegaConf
        sys.path[:0] = [str(UPSTREAM), str(ROOT / 'src')]
        # Persist the resolved training settings before adding the transient auth path.
        OmegaConf.save(cfg, run_dir / 'resolved_config.yaml')
        atomic_json(run_dir / 'runtime.json', {'ray': str(runtime), 'gpus': gpu_ids})
        with hint_credentials(cfg,args.command=='train' and not cfg.trainer.val_only,runtime,args.allow_api_cost):
            os.chdir(UPSTREAM)
            from verl.trainer.main_sdar import run_sdar
            import ray
            try:
                run_sdar(cfg)
                if args.command == 'train':
                    CheckpointStore(run_dir).verify(run_dir / 'checkpoints/global_step_150')
                atomic_json(run_dir / 'completion.json', {'status': 'completed', 'command': args.command,
                    'updates': 150 if args.command == 'train' else 0, 'binding': binding})
            finally:
                ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'eval', 'export'])
    parser.add_argument('--size', choices=['1.5b', '3b'], required=True)
    parser.add_argument('--gpus', default=os.environ.get('OPSD_GPUS', '0,1,2,3'))
    parser.add_argument('--cpus', type=int, default=int(os.environ.get('OPSD_CPUS', '32')))
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/searchqa')
    parser.add_argument('--retrieval-dir', type=Path, default=ROOT / 'data/retrieval')
    parser.add_argument('--search-url', default=os.environ.get('SEARCH_URL', 'http://127.0.0.1:8000/retrieve'))
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--model', type=Path, help='Verified Stage-1 Actor, or exported model for evaluation')
    parser.add_argument('--allow-api-cost', action='store_true',
                        help='Authorize paid calls only on Actor-triggered training steps')
    parser.add_argument('--eval-scope', choices=['full', 'subset'],
                        help='Independent eval only; default full. Training uses fixed subset then final full.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print native config without assets, GPU, API or training')
    args = parser.parse_args()
    if args.size != load('release.json')['size']:
        parser.error('Use the independently pinned repository for this model size')
    if args.cpus < 16:
        parser.error('Allocate at least16 logical CPUs;32 or more recommended for search/retrieval')
    if args.allow_api_cost and args.command != 'train':
        parser.error('--allow-api-cost is training only')
    if args.resume and args.command != 'train':
        parser.error('--resume is training only')
    if args.eval_scope is not None and args.command != 'eval':
        parser.error('--eval-scope is only for independent evaluation, not changing training validation')
    if args.command == 'export':
        verify()
        run_dir = (args.run_dir or ROOT / 'runs' / ('rethink_' + args.size)).resolve()
        with run_lock(run_dir):
            target = run_dir / 'hf150'
            if target.exists():
                raise RuntimeError('Export target exists; refusing overwrite')
            checkpoint = CheckpointStore(run_dir).recover()
            if checkpoint.name != 'global_step_150':
                raise RuntimeError('Export requires the verified final update150 checkpoint')
            subprocess.run([sys.executable, str(UPSTREAM / 'scripts/model_merger.py'), 'merge', '--backend', 'fsdp',
                '--local_dir', str(checkpoint / 'actor'), '--target_dir', str(target)],
                cwd=UPSTREAM, env=dict(os.environ, PYTHONPATH=str(UPSTREAM)), check=True)
            atomic_json(run_dir / 'export_manifest.json', {'step': 150,
                'files': {p.name: sha(p) for p in target.iterdir() if p.is_file()}})
    else:
        if args.command == 'eval' and not args.model:
            parser.error('Independent evaluation requires --model')
        launch(args)


if __name__ == '__main__':
    main()
