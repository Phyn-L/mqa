#!/usr/bin/env python
"""Convert the official PeerQA source JSONL files when available."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent; RAW=ROOT/'raw/peerqa/source'; OUT=ROOT/'standardized/peerqa'; OUT.mkdir(parents=True,exist_ok=True)

def main():
    papers_path=RAW/'papers.jsonl'; qa_path=RAW/'qa.jsonl'
    if not papers_path.exists() or not qa_path.exists():
        raise SystemExit(f'Expected {papers_path} and {qa_path}; run scripts/download_peerqa.sh first.')
    papers={}
    for line in papers_path.open(encoding='utf-8'):
        r=json.loads(line); papers.setdefault(str(r['paper_id']),[]).append(r)
    rows=[]
    for line in qa_path.open(encoding='utf-8'):
        q=json.loads(line); pid=str(q['paper_id']); ps=sorted(papers.get(pid,[]),key=lambda x:x.get('idx',0))
        context='\n\n'.join((f"## {p.get('last_heading')}\n" if p.get('last_heading') else '')+str(p.get('content','')) for p in ps if p.get('content'))
        answers=[x for x in [q.get('answer_free_form'),q.get('answer_free_form_augmented')] if x]
        evidence=q.get('answer_evidence_sent') or q.get('raw_answer_evidence') or []
        rows.append({'id':q.get('question_id'),'context':context,'question':q.get('question',''),'answers':answers,'answer_starts':[],'evidence':evidence,'metadata':{'paper_id':pid,'answerable':q.get('answerable'),'answerable_mapped':q.get('answerable_mapped'),'answer_evidence_mapped':q.get('answer_evidence_mapped'),'context_type':'paper_sentences','num_paper_sentences':len(ps)},'dataset':'peerqa','split':'all'})
    with (OUT/'all.jsonl').open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    print(OUT/'all.jsonl',len(rows))
if __name__=='__main__': main()
