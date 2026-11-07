"""Prepare pinned public Search-R1 assets or launch the original E5 server."""
import argparse
import gzip
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from common import ROOT, UPSTREAM, atomic_json, load, sha
from downloads import fetch
from setup import verify as verify_source


def verify(directory):
    directory = Path(directory)
    receipt = json.loads((directory / 'retrieval_receipt.json').read_text(encoding='utf-8'))
    if receipt['lock_sha256'] != sha(ROOT / 'configs/retrieval.json'):
        raise RuntimeError('Retrieval asset/encoder recipe changed')
    for name, record in receipt['files'].items():
        path = directory / name
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise RuntimeError('Retrieval asset hash differs: ' + name)
    return receipt


def prepare(directory):
    directory = Path(directory)
    lock = load('retrieval.json')
    if (directory / 'retrieval_receipt.json').exists():
        verify(directory)
        print('Existing complete retrieval assets verified.')
        return
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ('e5_Flat.index', 'wiki-18.jsonl')):
        raise RuntimeError('Uncertified complete retrieval output exists; preserve it and use a new directory')
    for group in ('index', 'corpus', 'encoder'):
        config = lock[group]
        dest = directory / ('encoder' if group == 'encoder' else 'downloads')
        for name, record in config['files'].items():
            fetch(config['repository'], config['revision'], name, dest, record,
                  repo_type='model' if group == 'encoder' else 'dataset')
    index = directory / 'e5_Flat.index'
    pending = index.with_suffix('.index.pending')
    with pending.open('wb') as output:
        for name in ('part_aa', 'part_ab'):
            with (directory / 'downloads' / name).open('rb') as source:
                shutil.copyfileobj(source, output, length=8 * 1024**2)
        output.flush()
        os.fsync(output.fileno())
    expected_bytes = sum(v['bytes'] for v in lock['index']['files'].values())
    if pending.stat().st_size != expected_bytes:
        raise RuntimeError('Merged native Flat index size differs')
    pending.replace(index)
    corpus = directory / 'wiki-18.jsonl'
    pending_corpus = corpus.with_suffix('.jsonl.pending')
    with gzip.open(directory / 'downloads/wiki-18.jsonl.gz', 'rb') as source, pending_corpus.open('wb') as output:
        shutil.copyfileobj(source, output, length=8 * 1024**2)
        output.flush()
        os.fsync(output.fileno())
    pending_corpus.replace(corpus)
    files = {'e5_Flat.index': {'bytes': index.stat().st_size, 'sha256': sha(index)},
             'wiki-18.jsonl': {'bytes': corpus.stat().st_size, 'sha256': sha(corpus)}}
    files.update({'encoder/' + name: record for name, record in lock['encoder']['files'].items()})
    atomic_json(directory / 'retrieval_receipt.json',
                {'schema': 1, 'lock_sha256': sha(ROOT / 'configs/retrieval.json'), 'files': files})
    verify(directory)
    print('Native index/corpus/E5 prepared; no GPU server or training started. Download parts retained.')


def serve(directory, gpus, port, index_cpu=False):
    verify_source()
    verify(directory)
    ids = [value.strip() for value in gpus.split(',')]
    if not ids or any(not value.isdigit() for value in ids) or len(ids) != len(set(ids)):
        raise ValueError('Specify allocated distinct GPU IDs; the unchanged E5 encoder requires CUDA')
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(ids), TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(UPSTREAM))
    source = UPSTREAM / 'examples/search/retriever/retrieval_server.py'
    sys.argv = [str(source), '--index_path', str(directory / 'e5_Flat.index'),
        '--corpus_path', str(directory / 'wiki-18.jsonl'), '--retriever_model', str(directory / 'encoder'),
        '--retriever_name', 'e5', '--topk', '3', '--port', str(port)]
    if not index_cpu:
        sys.argv.append('--faiss_gpu')
    runpy.run_path(str(source), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'data/retrieval')
    parser.add_argument('--prepare', action='store_true', help='Explicit large download: roughly 70GB compressed/index parts')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--gpus', default=os.environ.get('RETRIEVER_GPUS', '0,1,2,3'))
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--index-cpu', action='store_true', help='Explicit native CPU FAISS option; E5 still uses CUDA; not default')
    args = parser.parse_args()
    if not (args.prepare or args.verify or args.serve):
        parser.error('Choose --prepare, --verify or --serve')
    directory = args.directory.resolve()
    if args.prepare:
        prepare(directory)
    if args.verify:
        verify(directory)
        print('Pinned retrieval asset hashes passed.')
    if args.serve:
        serve(directory, args.gpus, args.port, args.index_cpu)


if __name__ == '__main__':
    main()
