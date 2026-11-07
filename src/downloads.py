"""Checksum-verified, revision-pinned public downloads; HF handles resume."""
from pathlib import Path
from common import sha


def fetch(repository, revision, filename, destination, record, repo_type='dataset'):
    from huggingface_hub import hf_hub_download
    destination = Path(destination)
    target = destination / filename
    if target.exists():
        if target.stat().st_size != record['bytes'] or sha(target) != record['sha256']:
            raise RuntimeError('Existing public asset differs; preserving it: ' + str(target))
        return target
    downloaded = Path(hf_hub_download(repository, filename=filename, revision=revision,
        repo_type=repo_type, local_dir=destination))
    if downloaded.stat().st_size != record['bytes'] or sha(downloaded) != record['sha256']:
        raise RuntimeError('Downloaded public asset checksum differs: ' + filename)
    return downloaded
