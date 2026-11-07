"""Transactional local storage around the unchanged native FSDP serializer.

The controller holds run_lock for every save/recovery/export. Only checkpoints
with this run's verified completion manifest are eligible for rotation.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import uuid
import zipfile

from common import WORLD_SIZE, sha

COMPLETE = 'CHECKPOINT_COMPLETE.json'
POINTER = 'latest_checkpointed_iteration.txt'
STEP_NAME = re.compile(r'global_step_([1-9][0-9]*)\Z')
LOGGER = logging.getLogger(__name__)


def _linked(path):
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def _sync_directory(path):
    if os.name == 'posix':
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_text(path, text):
    """Replace metadata without exposing a truncated pointer/manifest."""
    if _linked(path):
        raise RuntimeError('Refuse linked checkpoint metadata: ' + str(path))
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.pending')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _check_torch_archive(path):
    """Check the native torch.save ZIP, all member CRCs, without unpickling."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            roots = {name.split('/')[0] for name in names}
            if len(roots) != 1 or len(set(names)) != len(names):
                raise ValueError('Unexpected torch.save member layout')
            root = next(iter(roots))
            if root + '/data.pkl' not in names or root + '/version' not in names:
                raise ValueError('Missing torch.save metadata')
            if archive.testzip() is not None:
                raise ValueError('torch.save member CRC mismatch')
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as error:
        raise RuntimeError('Invalid native checkpoint archive: ' + str(path)) from error


class CheckpointStore:
    """Same-filesystem staging -> verify -> atomic publish -> rotate newest two.

    Calls are serialized by the launcher's run_lock, not an in-memory rank list.
    Partial, corrupt, foreign and linked directories are preserved, never pruned.
    """

    def __init__(self, run_dir):
        run_dir = Path(run_dir).absolute()
        if _linked(run_dir) or _linked(run_dir / 'checkpoints') or _linked(run_dir / 'binding.json'):
            raise RuntimeError('Refuse linked run/checkpoint directories or binding')
        self.run_dir = run_dir.resolve(strict=True)
        binding = json.loads((self.run_dir / 'binding.json').read_text(encoding='utf-8'))
        if (not isinstance(binding, dict) or binding.get('world_size') != WORLD_SIZE
                or binding.get('command') != 'train'):
            raise RuntimeError('Checkpoint storage requires this four-rank training binding')
        serialized = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False)
        self.binding_sha = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        self.root = self.run_dir / 'checkpoints'
        self.root.mkdir(exist_ok=True)

    def _step(self, path, staging=False):
        path = Path(path).absolute()
        match = STEP_NAME.fullmatch(path.name)
        if not match or _linked(path) or not path.is_dir():
            raise RuntimeError('Not an owned checkpoint directory: ' + str(path))
        if staging:
            parent = path.parent
            if (parent.parent != self.root or not parent.name.startswith('.saving-')
                    or _linked(parent) or parent.resolve().parent != self.root):
                raise RuntimeError('Checkpoint staging directory escaped the run')
        elif path.parent != self.root or path.resolve().parent != self.root:
            raise RuntimeError('Checkpoint directory escaped the run')
        return int(match[1])

    def _inventory(self, path):
        files = {}
        for directory, dirs, names in os.walk(path, followlinks=False):
            for name in dirs + names:
                item = Path(directory) / name
                if _linked(item) or not item.resolve().is_relative_to(path.resolve()):
                    raise RuntimeError('Linked/escaped checkpoint contents: ' + str(item))
                if not (item.is_dir() or stat.S_ISREG(item.stat().st_mode)):
                    raise RuntimeError('Non-regular checkpoint contents: ' + str(item))
            for name in names:
                item = Path(directory) / name
                relative = item.relative_to(path).as_posix()
                if relative != COMPLETE:
                    if item.stat().st_size == 0:
                        raise RuntimeError('Empty checkpoint file: ' + relative)
                    files[relative] = item
        required = {'data.pt', 'actor/config.json', 'actor/tokenizer.json',
                    'actor/tokenizer_config.json', 'actor/vocab.json', 'actor/merges.txt'}
        shards = {f'actor/{kind}_world_size_{WORLD_SIZE}_rank_{rank}.pt'
                  for rank in range(WORLD_SIZE) for kind in ('model', 'optim', 'extra_state')}
        if not (required | shards) <= files.keys():
            raise RuntimeError('Incomplete four-rank checkpoint: ' + str(path))
        if {name for name in files if name.endswith('.pt')} != shards | {'data.pt'}:
            raise RuntimeError('Unexpected rank/world-size/component layout: ' + str(path))
        for name, item in files.items():
            if name.endswith('.json'):
                value = json.loads(item.read_text(encoding='utf-8'))
                if not isinstance(value, dict):
                    raise RuntimeError('Invalid checkpoint JSON object: ' + name)
        return files

    def certify(self, path, *, staging=False):
        """Seal a completed native save, including legacy same-run saves on resume."""
        path = Path(path).absolute()
        step = self._step(path, staging)
        if (path / COMPLETE).exists() or _linked(path / COMPLETE):
            raise RuntimeError('Refuse to replace an existing completion manifest')
        files = self._inventory(path)
        records = {}
        for name, item in sorted(files.items()):
            # Native torch.save has closed the writer; explicitly flush its file.
            with item.open('r+b' if os.name == 'nt' else 'rb') as stream:
                os.fsync(stream.fileno())
            if name.endswith('.pt'):
                _check_torch_archive(item)
            records[name] = {'bytes': item.stat().st_size, 'sha256': sha(item)}
        for directory, _, _ in os.walk(path, topdown=False):
            _sync_directory(directory)
        receipt = {'schema': 1, 'step': step, 'world_size': WORLD_SIZE,
                   'binding_sha256': self.binding_sha, 'files': records}
        _atomic_text(path / COMPLETE, json.dumps(receipt, sort_keys=True, indent=2) + '\n')

    def verify(self, path, *, hashes=True):
        path = Path(path).absolute()
        step = self._step(path)
        files = self._inventory(path)
        manifest = path / COMPLETE
        if _linked(manifest):
            raise RuntimeError('Refuse linked completion manifest')
        receipt = json.loads(manifest.read_text(encoding='utf-8'))
        if (not isinstance(receipt, dict) or receipt.get('schema') != 1 or receipt.get('step') != step
                or receipt.get('world_size') != WORLD_SIZE
                or receipt.get('binding_sha256') != self.binding_sha):
            raise RuntimeError('Foreign checkpoint completion manifest: ' + str(path))
        records = receipt.get('files')
        if not isinstance(records, dict) or set(records) != set(files):
            raise RuntimeError('Checkpoint file inventory differs: ' + str(path))
        for name, item in files.items():
            record = records[name]
            if (not isinstance(record, dict) or record.get('bytes') != item.stat().st_size
                    or not re.fullmatch('[0-9a-f]{64}', str(record.get('sha256', '')))
                    or (hashes and sha(item) != record['sha256'])):
                raise RuntimeError('Checkpoint checksum/size mismatch: ' + name)
        return path

    def scan(self, *, adopt_legacy=False):
        complete = []
        for path in sorted(self.root.iterdir()):
            if not STEP_NAME.fullmatch(path.name):
                continue
            try:
                self._step(path)
                if adopt_legacy and not (path / COMPLETE).exists():
                    self.certify(path)
                complete.append(self.verify(path))
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                LOGGER.warning('Preserving unverified checkpoint %s: %s', path.name, error)
        return sorted(complete, key=lambda path: int(STEP_NAME.fullmatch(path.name)[1]))

    def _publish_pointer(self, step):
        _atomic_text(self.root / POINTER, str(step) + '\n')

    def _rotate(self, complete):
        """Only called with a fully SHA-verified disk scan under run_lock."""
        if not complete:
            raise RuntimeError('No complete checkpoint; refusing rotation or a fresh restart')
        self._publish_pointer(self._step(complete[-1]))
        for old in complete[:-2]:
            # Recheck exact target, binding, links and inventory immediately before deletion.
            self.verify(old, hashes=False)
            shutil.rmtree(old)
            _sync_directory(self.root)
            LOGGER.info('Removed verified superseded checkpoint %s', old.name)
        return complete[-1]

    def recover(self):
        """Repair a missing/stale pointer and keep two verified saves across restarts."""
        return self._rotate(self.scan(adopt_legacy=True))

    def save(self, step, write_native):
        if not isinstance(step, int) or step <= 0:
            raise ValueError('Expected a positive native update number')
        destination = self.root / f'global_step_{step}'
        if destination.exists() or _linked(destination):
            raise RuntimeError('Refuse to overwrite an existing checkpoint: ' + str(destination))
        staging = self.root / ('.saving-' + uuid.uuid4().hex)
        staging.mkdir()
        # Failure deliberately leaves the private staging tree for inspection.
        write_native(staging)
        candidate = staging / destination.name
        self.certify(candidate, staging=True)
        candidate.rename(destination)
        _sync_directory(self.root)
        # This re-reads every file SHA, including the new save, before any pruning.
        complete = self.scan()
        if destination not in complete:
            raise RuntimeError('New checkpoint failed verification; old checkpoints preserved')
        self._rotate(complete)
        # Only remove the small native staging tracker after successful commit.
        tracker = staging / POINTER
        if tracker.is_file() and not _linked(tracker):
            tracker.unlink()
        staging.rmdir()
        _sync_directory(self.root)
        return destination


class NativeCheckpointMixin:
    """Storage-only overrides; native save/load formats and fit stay unchanged."""

    def _checkpoint_store(self):
        store = CheckpointStore(os.environ['GRPO_RUN_DIR'])
        trainer = self.config.trainer
        if (Path(trainer.default_local_dir).resolve() != store.root
                or trainer.default_hdfs_dir is not None or self.use_critic
                or trainer.get('remove_previous_ckpt_in_save', False)):
            raise RuntimeError('Expected local GRPO-based checkpoint configuration')
        return store

    def _save_checkpoint(self):
        store = self._checkpoint_store()
        trainer = self.config.trainer
        step = int(self.global_steps)
        final = step == trainer.total_training_steps
        destination = store.root / f'global_step_{step}'
        if final and getattr(self, '_verified_final_save', None) == str(destination):
            # The final validation already saved this update in this controller.
            # Native fit calls save again afterward: verify, never overwrite.
            return store.verify(destination)
        native_save = super()._save_checkpoint

        def write_native(staging):
            previous = {key: trainer[key] for key in ('default_local_dir',
                        'max_actor_ckpt_to_keep', 'max_critic_ckpt_to_keep')}
            try:
                trainer.default_local_dir = str(staging)
                # Disable ONLY upstream's unsafe pre-save/in-memory pruning.
                trainer.max_actor_ckpt_to_keep = None
                trainer.max_critic_ckpt_to_keep = None
                native_save()
            finally:
                for key, value in previous.items():
                    trainer[key] = value

        checkpoint = store.save(step, write_native)
        if final:
            self._verified_final_save = str(checkpoint)
        return checkpoint

    def _load_checkpoint(self):
        if self.config.trainer.val_only:
            # Independent HF evaluation has an eval binding, not training shards.
            return super()._load_checkpoint()
        mode = self.config.trainer.resume_mode
        if mode == 'resume_path':
            self._checkpoint_store().verify(Path(self.config.trainer.resume_from_path))
        elif mode == 'auto':
            store = self._checkpoint_store()
            if any(STEP_NAME.fullmatch(path.name) for path in store.root.iterdir()):
                raise RuntimeError('Existing checkpoints require explicit --resume verification')
        return super()._load_checkpoint()
