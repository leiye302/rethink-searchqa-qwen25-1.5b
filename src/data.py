"""Pinned Search-R1 data using the unchanged upstream row preprocessor."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
from common import ROOT, UPSTREAM, atomic_json, digest, load, plain, question_id, sha
from downloads import fetch
from setup import verify as verify_source


def native_processor():
    """Execute the exact locked pure function without importing HDFS/CUDA."""
    import pandas as pd
    source = UPSTREAM / 'examples/data_preprocess/preprocess_search_r1_dataset.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('DEFAULT_SYSTEM_CONTENT', 'DEFAULT_USER_CONTENT_PREFIX'):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_single_row')
    scope = {'pd': pd, 'system_content': constants['DEFAULT_SYSTEM_CONTENT'],
             'user_content_prefix': constants['DEFAULT_USER_CONTENT_PREFIX']}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
    return scope['process_single_row']


def row_summary(frame):
    rows = [plain(v) for v in frame['env_kwargs']]
    counts = dict(sorted(Counter(v['data_source'] for v in rows).items()))
    identifiers = [question_id(v) for v in rows]
    return {'count': len(rows), 'source_counts': counts,
            'ordered_question_ids_sha256': digest(identifiers), 'question_ids': identifiers}


def verify(directory, require_subset=True):
    import pandas as pd
    directory = Path(directory)
    receipt = json.loads((directory / 'data_receipt.json').read_text(encoding='utf-8'))
    lock = load('data.json')
    if receipt['data_lock_sha256'] != sha(ROOT / 'configs/data.json'):
        raise RuntimeError('Dataset preparation recipe changed')
    if receipt['preprocessor_sha256'] != load('upstream.json')['unchanged_files'][
            'examples/data_preprocess/preprocess_search_r1_dataset.py']:
        raise RuntimeError('Preprocessor differs')
    for split in ('train', 'test'):
        path = directory / (split + '.parquet')
        record = receipt['files'][split]
        if sha(path) != record['sha256']:
            raise RuntimeError('Prepared Parquet checksum differs: ' + split)
        observed = row_summary(pd.read_parquet(path, columns=['env_kwargs']))
        expected = lock['splits'][split]
        if observed != record['rows'] or observed['count'] != expected['rows']:
            raise RuntimeError('Prepared question identities or counts differ: ' + split)
        if observed['source_counts'] != expected['source_counts']:
            raise RuntimeError('Benchmark source counts differ: ' + split)
        if split == 'test' and observed['ordered_question_ids_sha256'] != load('evaluation.json')['ordered_question_ids_sha256']:
            raise RuntimeError('Shared full-test question identities/order changed')
    if require_subset:
        from validation_subset import verify as verify_subset
        receipt = {**receipt, 'validation_subset': verify_subset(directory)}
    return receipt


def prepare(directory):
    import pandas as pd
    directory = Path(directory)
    verify_source()
    lock = load('data.json')
    if (directory / 'data_receipt.json').exists():
        verify(directory, require_subset=False)
        from validation_subset import prepare as prepare_subset
        prepare_subset(directory)
        verify(directory)
        print('Existing native data and fixed monitoring subset verified; no overwrite.')
        return
    if any((directory / (split + '.parquet')).exists() for split in ('train', 'test')):
        raise RuntimeError('Partial or foreign prepared data exists; preserve it and use a new directory')
    directory.mkdir(parents=True, exist_ok=True)
    processor = native_processor()
    receipt = {'schema': 1, 'data_lock_sha256': sha(ROOT / 'configs/data.json'),
        'preprocessor_sha256': load('upstream.json')['unchanged_files'][
            'examples/data_preprocess/preprocess_search_r1_dataset.py'], 'files': {}}
    raw_questions = {}
    for split in ('train', 'test'):
        source = fetch(lock['repository'], lock['revision'], split + '.parquet', directory / 'raw',
                       lock['splits'][split])
        raw = pd.read_parquet(source)
        # No resampling, relabeling, task exclusion or custom prompt replacement.
        processed = raw.apply(lambda row: processor(row, split, row.name), axis=1)
        rows = row_summary(processed)
        if rows['count'] != lock['splits'][split]['rows'] or rows['source_counts'] != lock['splits'][split]['source_counts']:
            raise RuntimeError('Pinned public dataset content differs: ' + split)
        target = directory / (split + '.parquet')
        temporary = target.with_suffix('.parquet.pending')
        processed.to_parquet(temporary, index=False)
        temporary.replace(target)
        receipt['files'][split] = {'sha256': sha(target), 'rows': rows}
        raw_questions[split] = {str(q).casefold().strip() for q in raw['question']}
        print(f'Prepared native {split}: {len(processed)} questions.', flush=True)
    receipt['exact_normalized_train_test_question_overlap'] = len(raw_questions['train'] & raw_questions['test'])
    receipt['overlap_policy'] = 'Report original split overlap; never silently filter an official reproduction.'
    atomic_json(directory / 'data_receipt.json', receipt)
    from validation_subset import prepare as prepare_subset
    prepare_subset(directory)
    verify(directory)
    print('All native questions preserved and verified. No GPU or training used.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'data/searchqa')
    parser.add_argument('--prepare', action='store_true', help='Download pinned QA Parquets and run native preprocessing')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not (args.prepare or args.verify):
        parser.error('Choose --prepare or --verify')
    if args.prepare:
        prepare(args.directory)
    if args.verify:
        receipt = verify(args.directory)
        summary = {s: {k:v for k,v in r['rows'].items() if k != 'question_ids'}
                   for s,r in receipt['files'].items()}
        summary['monitor_subset'] = {k:v for k,v in receipt['validation_subset']['rows'].items() if k != 'question_ids'}
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
