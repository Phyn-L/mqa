#!/usr/bin/env python
"""Fast context token-scale estimate: full character counts + sampled Qwen token/char ratios."""
import glob, hashlib, json, random, os
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent.parent
TOKENIZER=Path('/home/lz/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/tokenizer.json')
OUT=ROOT/'context_token_stats.json'; SAMPLE_PER_FILE=int(os.getenv('SAMPLE_PER_FILE','2000')); SEED=42

def stats(path):
    rng=random.Random(SEED); n=chars=words=empty=0; uniq_chars=0; seen=set(); sample=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            c=json.loads(line).get('context') or ''; n+=1; chars+=len(c); words+=len(c.split()); empty+=not bool(c.strip())
            h=hashlib.sha1(c.encode()).digest()
            if h not in seen: seen.add(h); uniq_chars+=len(c)
            if len(sample)<SAMPLE_PER_FILE: sample.append(c)
            else:
                j=rng.randrange(n)
                if j<SAMPLE_PER_FILE: sample[j]=c
    t=Tokenizer.from_file(str(TOKENIZER)); sample_chars=sum(map(len,sample)); sample_tokens=sum(len(x) for x in t.encode_batch(sample))
    ratio=sample_tokens/sample_chars if sample_chars else 0
    return {'file':str(Path(path).relative_to(ROOT)),'samples':n,'chars':chars,'whitespace_words':words,'empty_contexts':empty,'unique_contexts':len(seen),'unique_chars':uniq_chars,'sample_size':len(sample),'sample_qwen_tokens':sample_tokens,'sample_chars':sample_chars,'qwen_tokens_est':round(chars*ratio),'unique_qwen_tokens_est':round(uniq_chars*ratio),'mean_qwen_tokens_est':(chars*ratio/n if n else 0),'p50_chars':sorted(map(len,sample))[len(sample)//2] if sample else 0,'p95_chars':sorted(map(len,sample))[int(.95*len(sample))] if sample else 0}

def main():
    rows=[stats(f) for f in sorted(glob.glob(str(ROOT/'standardized/**/*.jsonl'),recursive=True))]
    groups={}
    for r in rows:
        ds=r['file'].split('/')[1]; g=groups.setdefault(ds,{k:0 for k in ['files','samples','chars','whitespace_words','empty_contexts','unique_contexts','unique_chars','qwen_tokens_est','unique_qwen_tokens_est']}); g['files']+=1
        for k in g:
            if k!='files': g[k]+=r[k]
    result={'method':'full character/whitespace counts; Qwen token/character ratio estimated from 2,000 reservoir samples per JSONL file','tokenizer':str(TOKENIZER),'files':rows,'datasets':groups}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(groups,ensure_ascii=False,indent=2)); print('written',OUT)
if __name__=='__main__': main()
