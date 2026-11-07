"""Portable paths and CPU-only contracts for the native Search-QA baseline."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.deps/sdar'
WORLD_SIZE = 4


def load(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def devices(spec):
    ids = [x.strip() for x in spec.split(',')]
    if len(ids) != WORLD_SIZE or len(set(ids)) != WORLD_SIZE or any(not x.isdigit() for x in ids):
        raise ValueError('Choose four distinct allocated GPU indices, e.g. 0,1,2,3')
    return ids


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    return value


def digest(value):
    payload = json.dumps(plain(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def question_id(env_kwargs):
    return digest({k: env_kwargs[k] for k in ('question', 'ground_truth', 'data_source')})
