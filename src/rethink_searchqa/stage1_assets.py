"""Verify SFT provenance. The 3B release contains Actor weights, not optimizer state."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request
from common import ROOT, digest, load, sha
from .shell import INTRO_DIGEST, MARKER, MARKER_ID, STYLE

ALLOWED = {'config.json', 'generation_config.json', 'added_tokens.json', 'tokenizer.json',
           'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja',
           'merges.txt', 'vocab.json', 'model.safetensors.index.json'}
SHARD = re.compile(r'model(?:-\d{5}-of-\d{5})?\.safetensors\Z')
LEGAL = {'LICENSE-QWEN-3B.txt', 'NOTICE.txt'}
OBJECTIVE = {'suffix_ce': 1.0, 'route_bce': .1, 'reasoning_kl': 0.0}


def read(file):
    return json.loads(Path(file).read_text(encoding='utf-8'))


def expected_model():
    return ROOT / load('release.json')['default_sft_actor']


def validate_files(path, expected):
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError('Missing model directory; prepare the documented Stage-1 Actor first')
    if any(n not in ALLOWED and not SHARD.fullmatch(n) for n in expected):
        raise ValueError('Unexpected file in the model-only manifest')
    actual = {f.name for f in path.iterdir()}
    if actual - LEGAL != set(expected):
        raise ValueError('Incomplete or unexpected Stage-1 model file set')
    for name,spec in expected.items():
        file = path / name
        if (file.is_symlink() or not file.is_file() or file.stat().st_size != spec['bytes']
                or sha(file) != spec['sha256']):
            raise ValueError('Stage-1 model checksum differs: ' + name)
    return check_hf_layout(path)


def check_hf_layout(path):
    config = read(path / 'config.json')
    if config.get('model_type') != 'qwen2' or config.get('auto_map'):
        raise ValueError('Expected Qwen2 weights and no remote code')
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        names = set(read(index)['weight_map'].values())
        if not names or any(not SHARD.fullmatch(n) or not (path/n).is_file() for n in names):
            raise ValueError('Missing indexed model shards')
    elif not (path / 'model.safetensors').is_file():
        raise ValueError('Missing model weights')
    tokens = read(path / 'tokenizer.json')
    if not any(t.get('content') == MARKER and t.get('id') == MARKER_ID
               and t.get('special') is True for t in tokens['added_tokens']):
        raise ValueError('RETHINK is not the registered expected special token')
    return config


def verify(path, size, *, evaluation=False):
    path = Path(path)
    if evaluation:
        cfg = check_hf_layout(path)
        if cfg.get('hidden_size') != {'3b':2048,'1.5b':1536}[size]:
            raise ValueError('Evaluation model size differs')
        return {'evaluation_model_config_sha256':sha(path/'config.json')}
    if size == '3b':
        manifest = load('sft800_3b.json')
        if manifest['schema'] != 'searchqa-wrapped-sft800-model-v1':
            raise ValueError('Expected the fixed released 3B Actor')
    else:
        manifest = read(path.parent / 'stage1_manifest.json')
        if (manifest.get('schema') != 'searchqa-wrapped-sft800-local-v1'
                or manifest.get('data_lock_sha256') != sha(ROOT/'configs/sft_data.json')):
            raise ValueError('Run the documented 1.5B Route-SFT first')
    model = load('models.json')[size]
    if (manifest['size'] != size or manifest['model_id'] != model['model_id']
            or manifest['revision'] != model['revision'] or manifest['completed_updates'] != 800
            or manifest['protocol'] != STYLE or manifest['intro_digest'] != INTRO_DIGEST
            or manifest['marker_id'] != MARKER_ID or manifest['loss'] != OBJECTIVE
            or manifest['data_mix'] != {'good':4000,'bad':2400}):
        raise ValueError('Refusing a different size, base, data/recipe, protocol or intermediate Actor')
    cfg = validate_files(path, manifest['files'])
    if cfg.get('hidden_size') != {'3b':2048,'1.5b':1536}[size]:
        raise ValueError('Stage-1 model size differs')
    return {'stage1_model_manifest_digest':digest(manifest),'stage1_updates':800}


def fetch_asset(url, path, spec):
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']:
            raise ValueError('Existing download differs; choose a fresh model directory: '+path.name)
        return
    pending=path.with_name(path.name+'.pending')
    h=hashlib.sha256()
    size=0
    with urllib.request.urlopen(url,timeout=60) as stream, pending.open('xb') as out:
        while block:=stream.read(4*1024**2):
            size+=len(block)
            if size>spec['bytes']:
                raise ValueError('Asset exceeds its published byte count')
            h.update(block); out.write(block)
    if size!=spec['bytes'] or h.hexdigest()!=spec['sha256']:
        raise ValueError('Asset checksum mismatch: '+path.name)
    pending.rename(path)
    print('Verified',path.name,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-3b',action='store_true')
    p.add_argument('--accept-model-license',action='store_true')
    p.add_argument('--model',type=Path)
    p.add_argument('--size',choices=['3b','1.5b'],default=load('release.json')['size'])
    a=p.parse_args()
    path=a.model or expected_model()
    if a.download_3b:
        if a.size!='3b' or not a.accept_model_license:
            p.error('Read and accept the pinned Qwen Research License to download the 3B Actor')
        m=load('sft800_3b.json')
        required=sum(s['bytes'] for n,s in {**m['files'],**m['notices']}.items() if not (path/n).is_file())
        path.mkdir(parents=True,exist_ok=True)
        if path.is_symlink() or shutil.disk_usage(path).free < required:
            raise ValueError('Use a real directory with room for the remaining release assets')
        base=f"https://github.com/{m['repository']}/releases/download/{m['tag']}"
        for name,spec in {**m['notices'],**m['files']}.items():
            fetch_asset(f'{base}/{name}',path/name,spec)
    print(json.dumps(verify(path,a.size),sort_keys=True))


if __name__=='__main__':
    main()
