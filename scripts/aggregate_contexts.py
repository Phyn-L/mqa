#!/usr/bin/env python
"""Aggregate standardized QA rows by context.

MS MARCO v1.1 and v2.1 intentionally share one output namespace so that
cross-version duplicate contexts are merged while source versions remain in
each QA pair's metadata.
"""
import argparse, hashlib, json, re, sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'standardized'
OUTPUT = ROOT / 'aggregated'
DB = ROOT / '.context_aggregation.sqlite3'

def target_dataset(name):
    return 'ms_marco' if name.startswith('ms_marco_') else name

def source_version(name):
    m = re.search(r'v(\d+_\d+)', name)
    return m.group(1).replace('_', '.') if m else None

def init_db(conn):
    conn.executescript('''
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = NORMAL;
      CREATE TABLE IF NOT EXISTS contexts (
        dataset TEXT NOT NULL,
        context_hash TEXT NOT NULL,
        context TEXT NOT NULL,
        qa_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(dataset, context_hash)
      );
      CREATE TABLE IF NOT EXISTS qa_pairs (
        dataset TEXT NOT NULL,
        context_hash TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        qa_json TEXT NOT NULL,
        PRIMARY KEY(dataset, context_hash, ordinal)
      );
    ''')

def aggregate(input_dir=INPUT, output_dir=OUTPUT, db_path=DB):
    files = sorted(input_dir.glob('**/*.jsonl'))
    if not files:
        raise SystemExit(f'no JSONL files found under {input_dir}')
    if db_path.exists(): db_path.unlink()
    conn = sqlite3.connect(db_path)
    init_db(conn)
    counts = defaultdict(lambda: {'qa_rows': 0, 'contexts': 0, 'cross_version_contexts': 0, 'cross_version_qa_rows': 0})
    for path in files:
        for line in path.open(encoding='utf-8'):
            row = json.loads(line)
            original_dataset = row.get('dataset') or path.parts[-2]
            dataset = target_dataset(original_dataset)
            context = row.get('context') or ''
            context_hash = hashlib.sha256(context.encode('utf-8')).hexdigest()
            qa = dict(row)
            qa.pop('context', None)
            qa['source_dataset'] = original_dataset
            qa['source_split'] = row.get('split')
            version = source_version(original_dataset)
            if version:
                qa['source_version'] = version
            qa_json = json.dumps(qa, ensure_ascii=False, separators=(',', ':'))
            cur = conn.execute('SELECT qa_count FROM contexts WHERE dataset=? AND context_hash=?', (dataset, context_hash))
            found = cur.fetchone()
            if found is None:
                conn.execute('INSERT INTO contexts(dataset,context_hash,context,qa_count) VALUES(?,?,?,0)', (dataset, context_hash, context))
                ordinal = 0
            else:
                ordinal = found[0]
            conn.execute('INSERT INTO qa_pairs(dataset,context_hash,ordinal,qa_json) VALUES(?,?,?,?)', (dataset, context_hash, ordinal, qa_json))
            conn.execute('UPDATE contexts SET qa_count=qa_count+1 WHERE dataset=? AND context_hash=?', (dataset, context_hash))
            counts[dataset]['qa_rows'] += 1
        conn.commit()

    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, in conn.execute('SELECT DISTINCT dataset FROM contexts ORDER BY dataset'):
        out_dir = output_dir / dataset; out_dir.mkdir(exist_ok=True)
        out = out_dir / 'contexts.jsonl'
        qa_rows = contexts = 0
        with out.open('w', encoding='utf-8') as f:
            query = 'SELECT context_hash,context,qa_count FROM contexts WHERE dataset=? ORDER BY context_hash'
            for h, context, qa_count in conn.execute(query, (dataset,)):
                pairs=[]; versions=set(); source_datasets=set(); source_splits=set()
                for (qa_json,) in conn.execute('SELECT qa_json FROM qa_pairs WHERE dataset=? AND context_hash=? ORDER BY ordinal', (dataset,h)):
                    qa=json.loads(qa_json); pairs.append(qa); qa_rows+=1
                    source_datasets.add(qa.get('source_dataset')); source_splits.add(qa.get('source_split'))
                    if qa.get('source_version'): versions.add(qa['source_version'])
                rec={'context_id':h,'context':context,'qa_pairs':pairs,'num_qa_pairs':qa_count,'source_datasets':sorted(x for x in source_datasets if x),'source_splits':sorted(x for x in source_splits if x),'source_versions':sorted(versions)}
                f.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n'); contexts+=1
        counts[dataset]['contexts']=contexts
        counts[dataset]['qa_rows_exported']=qa_rows
        if dataset == 'ms_marco':
            cross_ctx=cross_qa=0
            for h, in conn.execute('SELECT context_hash FROM contexts WHERE dataset=?', (dataset,)):
                vs={r[0] for r in conn.execute("SELECT json_extract(qa_json,'$.source_version') FROM qa_pairs WHERE dataset=? AND context_hash=?", (dataset,h)) if r[0]}
                if len(vs)>1:
                    cross_ctx+=1
                    cross_qa += conn.execute('SELECT COUNT(*) FROM qa_pairs WHERE dataset=? AND context_hash=?', (dataset,h)).fetchone()[0]
            counts[dataset]['cross_version_contexts']=cross_ctx; counts[dataset]['cross_version_qa_rows']=cross_qa
    (ROOT/'context_aggregation_stats.json').write_text(json.dumps({'method':'exact context string; MS MARCO v1.1/v2.1 share one namespace','datasets':dict(counts)},ensure_ascii=False,indent=2),encoding='utf-8')
    conn.close(); db_path.unlink(missing_ok=True)

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=INPUT); ap.add_argument('--output',type=Path,default=OUTPUT); ap.add_argument('--db',type=Path,default=DB); args=ap.parse_args(); aggregate(args.input,args.output,args.db); print('written',args.output)
