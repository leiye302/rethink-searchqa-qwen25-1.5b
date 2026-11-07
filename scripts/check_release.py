"""Audit the explicit public allowlist, file/history privacy and source locks."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    '.env.example', '.gitattributes', '.gitignore', '.github/workflows/check.yml',
    'LICENSE', 'README.md', 'THIRD_PARTY.md', 'docs/EVALUATION.md', 'docs/UPSTREAM_DIFF.md',
    'configs/models.json', 'configs/release.json', 'configs/experiment.json',
    'configs/data.json', 'configs/evaluation.json', 'configs/retrieval.json', 'configs/validation_subset.json',
    'configs/grpo_comparison.json', 'tests/test_sdar.py', 'tests/test_grpo_parity.py',
    'configs/upstream.json', 'configs/requirements.txt', 'configs/requirements-cpu.txt',
    'configs/requirements-retriever.txt', 'patches/runtime.patch',
    'scripts/check_release.py', 'scripts/train.sh', 'scripts/eval.sh', 'scripts/start_retriever.sh',
    'src/common.py', 'src/setup.py', 'src/downloads.py', 'src/data.py', 'src/run.py',
    'src/retrieval.py', 'src/checkpoint_io.py', 'src/evaluation_contract.py',
    'src/grpo_eval.py', 'src/env_smoke.py', 'src/report.py', 'src/validation_subset.py',
    'tests/test_checkpoint_io.py', 'tests/test_contract.py', 'tests/test_data.py',
    'tests/test_evaluation.py', 'tests/test_native_env.py', 'tests/test_subset.py',
    'tests/test_native_validation_boundary.py',
    'configs/sft800_3b.json', 'tests/test_rethink.py',
    'src/rethink_searchqa/__init__.py', 'src/rethink_searchqa/shell.py',
    'src/rethink_searchqa/action_wrapper.py', 'src/rethink_searchqa/history_display.py',
    'src/rethink_searchqa/protocol.py', 'src/rethink_searchqa/stage1_assets.py',
    'src/rethink_searchqa/hint_api.py', 'src/rethink_searchqa/hints.py',
    'src/rethink_searchqa/rethink_api_protocol.py', 'src/rethink_searchqa/rethink_benchmark_contract.py',
    'src/rethink_searchqa/trainer.py',
    'configs/sft_data.json', 'configs/requirements-sft.txt', 'docs/STAGE1.md',
    'src/rethink_searchqa/sft.py', 'src/rethink_searchqa/sft_eval.py',
    'src/rethink_searchqa/gpu_resident.py', 'tests/test_sft.py',
    'src/sft_reference/legacy/rethink_token.py', 'src/sft_reference/legacy/stage1_training.py',
    'src/sft_reference/route_bce/__init__.py', 'src/sft_reference/route_bce/core.py',
}
PRIVATE = [
    re.compile(r'[A-Za-z]:[/\\](?:Users[/\\]|app[/\\]|mnt[/\\])', re.I),
    re.compile(r'/mnt/[a-z0-9_.-]+/(?:test/)?VLA_test(?:/|\b)', re.I),
    re.compile(r'connect\.[a-z0-9]+\.seetacloud\.com', re.I),
    re.compile(r'(?:oc_sk_|github_pat_|ghp_)[A-Za-z0-9_-]{18,}'),
    re.compile(r'\bsk-[A-Za-z0-9_-]{24,}'),
    re.compile(r'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----'),
]


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True).stdout


def audit():
    files = set(git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().strip('\0').split('\0'))
    if files != ALLOWED:
        raise RuntimeError('Public allowlist differs: ' + repr(sorted(files ^ ALLOWED)))
    for name in sorted(files):
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError('Linked release file: ' + name)
        if any(pattern.search(path.read_text(encoding='utf-8')) for pattern in PRIVATE):
            raise RuntimeError('Private material in public file: ' + name)
    refs = git('rev-list', '--objects', '--all').decode().splitlines()
    blobs = 0
    for row in refs:
        key = row.split(' ', 1)[0]
        if git('cat-file', '-t', key).strip() != b'blob':
            continue
        payload = git('cat-file', 'blob', key)
        if len(payload) > 2 * 1024**2 or any(pattern.search(payload.decode('utf-8', errors='replace')) for pattern in PRIVATE):
            raise RuntimeError('Unexpected large or private historical blob')
        blobs += 1
    sys.path.insert(0, str(ROOT / 'src'))
    from setup import verify
    verify()
    print(f'PASS: {len(files)} allowlisted files; {blobs} historical blobs; native source locks.')


if __name__ == '__main__':
    audit()
