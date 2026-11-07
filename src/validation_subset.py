"""Fixed, source-balanced monitoring questions; final full test stays untouched."""
from collections import Counter
import json
from pathlib import Path

from common import ROOT, atomic_json, digest, load, plain, question_id, sha

SCHEMA = 'searchqa-balanced-monitor-v1'
MANIFEST = 'validation_subset.json'
PARQUET = 'validation_subset.parquet'
RECEIPT = 'validation_subset_receipt.json'


def normalized_question(value):
    return str(value).casefold().strip()


def select_indices(test_rows, train_questions, sources, per_source=64, seed=2026):
    """Seeded hash sampling without replacement; no scores or model outputs."""
    blocked = {normalized_question(q) for q in train_questions}
    pools = {source: [] for source in sorted(sources)}
    for index, raw in enumerate(test_rows):
        row = plain(raw)
        source = row['data_source']
        if source not in pools:
            raise RuntimeError('Unexpected test source')
        text = normalized_question(row['question'])
        if text not in blocked:
            rank = digest({'seed': seed, 'source': source, 'question': text})
            pools[source].append((rank, index, text))
    for pool in pools.values():
        pool.sort()
    positions = {source: 0 for source in pools}
    selected, used_texts = [], set()
    for _ in range(per_source):
        for source, pool in pools.items():
            while positions[source] < len(pool) and pool[positions[source]][2] in used_texts:
                positions[source] += 1
            if positions[source] == len(pool):
                raise RuntimeError('Not enough unique, non-overlapping questions in source: ' + source)
            _, index, text = pool[positions[source]]
            positions[source] += 1
            used_texts.add(text)
            selected.append(index)
    return selected


def make_manifest(test_rows, train_questions, per_source=64, seed=2026):
    data = load('data.json')
    evaluation = load('evaluation.json')
    sources = data['splits']['test']['source_counts']
    identities = [question_id(plain(row)) for row in test_rows]
    if len(identities) != data['splits']['test']['rows'] or digest(identities) != evaluation['ordered_question_ids_sha256']:
        raise RuntimeError('Manifest must be made from the exact pinned full test order')
    indices = select_indices(test_rows, train_questions, sources, per_source, seed)
    rows = [{'test_index': i, 'data_source': str(test_rows[i]['data_source']),
             'question_id': identities[i]} for i in indices]
    return {'schema': SCHEMA, 'dataset': data['repository'], 'revision': data['revision'],
        'full_test_raw_sha256': data['splits']['test']['sha256'],
        'full_test_question_ids_sha256': digest(identities), 'seed': seed,
        'questions_per_source': per_source, 'questions': len(rows),
        'source_counts': {name: per_source for name in sorted(sources)},
        'selection': 'seeded question-text SHA256, source-balanced without replacement, round-robin order',
        'training_question_overlap_policy': 'exclude str(question).casefold().strip() matches; no semantic-disjointness claim',
        'subset_question_texts_unique': True,
        'ordered_question_ids_sha256': digest([r['question_id'] for r in rows]),
        'selection_sha256': digest(rows), 'rows': rows}


def manifest():
    value = load(MANIFEST)
    evaluation = load('evaluation.json')
    sources = load('data.json')['splits']['test']['source_counts']
    if (value.get('schema') != SCHEMA or value.get('questions') != 448 or
            value.get('questions_per_source') != 64 or value.get('seed') != 2026 or
            value.get('source_counts') != {name: 64 for name in sorted(sources)}):
        raise RuntimeError('Shared monitoring subset recipe differs')
    rows = value['rows']
    if (len(rows) != value['questions'] or len({r['test_index'] for r in rows}) != len(rows) or
            len({r['question_id'] for r in rows}) != len(rows) or digest(rows) != value['selection_sha256'] or
            digest([r['question_id'] for r in rows]) != value['ordered_question_ids_sha256'] or
            dict(Counter(r['data_source'] for r in rows)) != value['source_counts'] or
            value['full_test_question_ids_sha256'] != evaluation['ordered_question_ids_sha256']):
        raise RuntimeError('Subset identity/order/source coverage differs')
    return value


def row_summary(frame):
    rows = [plain(row) for row in frame['env_kwargs']]
    identities = [question_id(row) for row in rows]
    return {'count': len(rows), 'source_counts': dict(sorted(Counter(r['data_source'] for r in rows).items())),
            'question_ids': identities, 'ordered_question_ids_sha256': digest(identities)}


def verify(directory):
    import pandas as pd
    directory = Path(directory)
    locked = manifest()
    record = json.loads((directory / RECEIPT).read_text(encoding='utf-8'))
    full = json.loads((directory / 'data_receipt.json').read_text(encoding='utf-8'))
    if (record.get('schema') != SCHEMA or record.get('manifest_sha256') != sha(ROOT / 'configs' / MANIFEST) or
            record.get('full_test_prepared_sha256') != full['files']['test']['sha256'] or
            record.get('subset_sha256') != sha(directory / PARQUET)):
        raise RuntimeError('Monitoring subset content/manifest binding differs')
    frame = pd.read_parquet(directory / PARQUET)
    observed = row_summary(frame)
    wanted = {'count': locked['questions'], 'source_counts': locked['source_counts'],
              'question_ids': [r['question_id'] for r in locked['rows']],
              'ordered_question_ids_sha256': locked['ordered_question_ids_sha256']}
    if observed != wanted or record.get('rows') != wanted or record.get('source_test_indices') != [r['test_index'] for r in locked['rows']]:
        raise RuntimeError('Monitoring subset question order/coverage differs')
    if [plain(v)['index'] for v in frame['extra_info']] != list(range(len(frame))):
        raise RuntimeError('Subset local input indices must be consecutive')
    original = pd.read_parquet(directory / 'test.parquet').iloc[record['source_test_indices']].copy().reset_index(drop=True)
    original['extra_info'] = [{**plain(value), 'index': i} for i, value in enumerate(original['extra_info'])]
    if digest(plain(frame.to_dict('records'))) != digest(plain(original.to_dict('records'))):
        raise RuntimeError('Subset native prompts, answers or other row fields were changed')
    return record


def prepare(directory):
    import pandas as pd
    directory = Path(directory)
    locked = manifest()
    if (directory / RECEIPT).exists():
        verify(directory)
        return
    target = directory / PARQUET
    pending = target.with_suffix('.parquet.pending')
    if target.exists() or pending.exists():
        raise RuntimeError('Partial or foreign subset exists; refusing overwrite')
    full = pd.read_parquet(directory / 'test.parquet')
    train = pd.read_parquet(directory / 'train.parquet', columns=['env_kwargs'])
    expected = make_manifest(list(full['env_kwargs']), [row['question'] for row in train['env_kwargs']])
    if expected != locked:
        raise RuntimeError('Pinned data cannot reproduce the shared subset selection')
    indices = [row['test_index'] for row in locked['rows']]
    subset = full.iloc[indices].copy().reset_index(drop=True)
    # Preserve native prompt, env_kwargs and verifier targets exactly. Only the
    # evaluation-local bookkeeping index changes; origin indices stay in manifest.
    subset['extra_info'] = [{**plain(value), 'index': i} for i, value in enumerate(subset['extra_info'])]
    subset.to_parquet(pending, index=False)
    pending.replace(target)
    full_receipt = json.loads((directory / 'data_receipt.json').read_text(encoding='utf-8'))
    atomic_json(directory / RECEIPT, {'schema': SCHEMA,
        'manifest_sha256': sha(ROOT / 'configs' / MANIFEST),
        'full_test_prepared_sha256': full_receipt['files']['test']['sha256'],
        'subset_sha256': sha(target), 'source_test_indices': indices, 'rows': row_summary(subset),
        'known_exact_training_question_overlap': 0,
        'native_row_change': 'extra_info.index only; prompts/env_kwargs/answers unchanged'})
    verify(directory)
    print('Fixed source-balanced monitoring subset prepared: 7 x 64 = 448 questions.', flush=True)
