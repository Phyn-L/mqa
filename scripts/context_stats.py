#!/usr/bin/env python
"""Estimate context-token scale for standardized JSONL files."""
import csv, glob, hashlib, json, os, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = Path('/home/lz/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/tokenizer.json')
OUT = ROOT / 'context_token_stats.json'
_local = __import__('threading').local()

def tok():
    if not hasattr(_local, 'tokenizer'):
        _local.tokenizer = Tokenizer.from_file(str(TOKENIZER))
    return _local.tokenizer

def percentile(sample, q):
    if not sample: return 0
    a = sorted(sample); return a[min(len(a)-1, int(q*(len(a)-1)))]

def one(path):
    total = chars = words = tokens = empty = 0
    seen = {}
    sample = []
    for line in open(path, encoding='utf-8'):
        row = json.loads(line); c = row.get('context') or ''
        n = len(tok().encode(c).ids); total += 1; chars += len(c); words += len(c.split()); tokens += n; empty += not bool(c.strip())
        key = hashlib.sha1(c.encode('utf-8')).digest()
        seen.setdefault(key, n)
        if len(sample) < 10000: sample.append(n)
        else:
            j = random.randrange(total)
            if j < len(sample): sample[j] = n
    return {'file': str(Path(path).relative_to(ROOT)), 'samples': total, 'chars': chars, 'whitespace_words': words, 'qwen_tokens': tokens, 'empty_contexts': empty, 'unique_contexts': len(seen), 'unique_qwen_tokens': sum(seen.values()), 'mean_qwen_tokens': tokens/total if total else 0, 'p50_qwen_tokens': percentile(sample,.50), 'p95_qwen_tokens': percentile(sample,.95), 'max_qwen_tokens': max(sample) if sample else 0}

def main():
    files = sorted(glob.glob(str(ROOT/'standardized/**/*.jsonl'), recursive=True))
    rows=[]
    with ThreadPoolExecutor(max_workers=min(8, len(files))) as ex:
        futs=[ex.submit(one,f) for f in files]
        for f in as_completed(futs): rows.append(f.result())
    rows.sort(key=lambda x:x['file'])
    groups={}
    for r in rows:
        ds=r['file'].split('/')[1]; g=groups.setdefault(ds, {k:0 for k in ['files','samples','chars','whitespace_words','qwen_tokens','empty_contexts','unique_contexts','unique_qwen_tokens']})
        g['files']+=1
        for k in g:
            if k!='files': g[k]+=r[k]
    result={'tokenizer': str(TOKENIZER), 'note':'unique totals are deduplicated within each JSONL file and summed across files; QA-expanded totals count repeated contexts per question.', 'files':rows, 'datasets':groups}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(groups, ensure_ascii=False, indent=2))
    print('written', OUT)

if __name__ == '__main__': main()
