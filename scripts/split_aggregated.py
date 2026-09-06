#!/usr/bin/env python
"""Write split-specific views of aggregated context records.

The original contexts.jsonl remains the canonical all-split aggregate. Split
views retain only QA pairs belonging to one normalized split. A context can
therefore occur in more than one view when the source dataset has cross-split
duplicates; this is reported in split_aggregation_stats.json.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent; AGG=ROOT/'aggregated'; SPLITS=('train','validation','test')

def normalize_split(value):
    s=str(value or '').lower()
    if 'train' in s: return 'train'
    if 'valid' in s or 'dev' in s: return 'validation'
    if 'test' in s: return 'test'
    # SciDQA's multi-document subset has no train/dev/test split; keep it in
    # train rather than silently dropping it from the requested views.
    if s == 'multidoc': return 'train'
    return None

def main():
    report={}
    for src in sorted(AGG.glob('*/contexts.jsonl')):
        dataset=src.parent.name; by_split=defaultdict(list); context_splits=defaultdict(set); total=0
        for line in src.open(encoding='utf-8'):
            rec=json.loads(line); per=defaultdict(list)
            for qa in rec.get('qa_pairs',[]):
                split=normalize_split(qa.get('source_split') or qa.get('split'))
                if split in SPLITS: per[split].append(qa); context_splits[rec['context_id']].add(split)
            for split,pairs in per.items():
                r={'context_id':rec['context_id'],'context':rec['context'],'qa_pairs':pairs,'num_qa_pairs':len(pairs),'source_datasets':sorted({q.get('source_dataset') for q in pairs if q.get('source_dataset')}),'source_splits':sorted({q.get('source_split') for q in pairs if q.get('source_split')}),'source_versions':sorted({q.get('source_version') for q in pairs if q.get('source_version')})}
                by_split[split].append(r); total += len(pairs)
        out_counts={};
        for split in SPLITS:
            out=src.parent/f'{split}.jsonl'
            with out.open('w',encoding='utf-8') as f:
                for r in by_split.get(split,[]): f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
            out_counts[split]={'contexts':len(by_split.get(split,[])),'qa_pairs':sum(r['num_qa_pairs'] for r in by_split.get(split,[]))}
        overlap=sum(len(s)>1 for s in context_splits.values())
        report[dataset]={'source_qa_pairs':total,'split_views':out_counts,'contexts_in_multiple_split_views':overlap}
    (ROOT/'split_aggregation_stats.json').write_text(json.dumps({'method':'filter QA pairs by normalized source split; contexts may repeat across views when source data crosses splits','datasets':report},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
