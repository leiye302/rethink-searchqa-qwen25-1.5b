"""Fetch pinned public SDAR and optional base weights; never start training."""
import argparse
import hashlib
import subprocess
import time
import uuid
from pathlib import Path
from common import ROOT, UPSTREAM, atomic_json, load, sha


def git(path, *args):
    command = ['git', '-c', 'core.autocrlf=false', '-C', str(path), *args]
    attempts = 3 if args and args[0] == 'fetch' else 1
    for attempt in range(attempts):
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
        if result.returncode == 0:
            return result.stdout.strip()
        if attempt + 1 < attempts:
            print('Public source transfer interrupted; retrying the same pinned revision.', flush=True)
            time.sleep(attempt + 1)
        else:
            result.check_returncode()


def verify():
    lock = load('upstream.json')
    if git(UPSTREAM, 'rev-parse', 'HEAD') != lock['commit']:
        raise RuntimeError('SDAR revision differs')
    changes = set(git(UPSTREAM, 'diff', '--name-only', 'HEAD', '--').splitlines())
    if changes != set(lock['patched_files']):
        raise RuntimeError('Unexpected native source changes: ' + ', '.join(sorted(changes)))
    if sha(ROOT / 'patches/runtime.patch') != lock['patch_sha256']:
        raise RuntimeError('Runtime patch differs')
    git(UPSTREAM, 'apply', '--reverse', '--check', str(ROOT / 'patches/runtime.patch'))
    for group in ('unchanged_files', 'patched_files'):
        for name, expected in lock[group].items():
            content = (UPSTREAM / name).read_bytes().replace(b'\r\n', b'\n')
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError('Native source hash mismatch: ' + name)
    return lock


def framework(source=None):
    lock = load('upstream.json')
    if UPSTREAM.exists():
        verify()
        print('Pinned native SDAR/Search-QA verified.')
        return
    UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
    staging = UPSTREAM.parent / ('sdar-' + uuid.uuid4().hex[:8])
    if source:
        git(ROOT, 'clone', '--no-checkout', '--no-hardlinks', str(source.resolve()), str(staging))
        git(staging, 'checkout', '--detach', lock['commit'])
        git(staging, 'remote', 'set-url', 'origin', lock['repository'])
    else:
        staging.mkdir()
        git(staging, 'init')
        git(staging, 'remote', 'add', 'origin', lock['repository'])
        git(staging, 'fetch', '--depth', '1', 'origin', lock['commit'])
        git(staging, 'checkout', '--detach', 'FETCH_HEAD')
    patch = ROOT / 'patches/runtime.patch'
    if sha(patch) != lock['patch_sha256']:
        raise RuntimeError('Unexpected patch hash')
    git(staging, 'apply', '--check', str(patch))
    git(staging, 'apply', str(patch))
    staging.rename(UPSTREAM)
    verify()
    print('Public upstream prepared; no GPU, paid API or training started.')


def download(size):
    from huggingface_hub import snapshot_download
    model = load('models.json')[size]
    target = ROOT / 'models' / model['model_id'].split('/')[-1]
    snapshot_download(model['model_id'], revision=model['revision'], local_dir=target,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE', 'README.md'])
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(target.iterdir())
             if p.is_file() and p.name != 'download_receipt.json'}
    atomic_json(target / 'download_receipt.json', dict(model, files=files))
    print('Pinned base model downloaded and hashed:', model['model_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', action='store_true')
    parser.add_argument('--upstream-source', type=Path, help='Optional read-only Git mirror; not required')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--download-model', choices=['1.5b', '3b'])
    args = parser.parse_args()
    if args.upstream_source and not args.framework:
        parser.error('--upstream-source requires --framework')
    if not (args.framework or args.verify or args.download_model):
        parser.error('Choose --framework, --verify or --download-model')
    if args.framework:
        framework(args.upstream_source)
    if args.verify:
        verify()
        print('Native algorithm, environment, prompt, reward, retrieval and preprocessing locks passed.')
    if args.download_model:
        download(args.download_model)


if __name__ == '__main__':
    main()
