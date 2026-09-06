#!/usr/bin/env python
"""Normalize Qasper and SciDQA while retaining paper/evidence metadata."""
import json, pickle, re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent; RAW=ROOT/'raw'; OUT=ROOT/'standardized'

def write_rows(ds, split, rows):
    p=OUT/ds; p.mkdir(parents=True,exist_ok=True); out=p/f'{split}.jsonl'
    with out.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    print(out, len(rows))

def paper_context(paper):
    parts=[f"# {paper.get('title') or ''}" ]
    if paper.get('abstract'): parts.append('## Abstract\n'+str(paper['abstract']))
    for sec in paper.get('full_text',[]):
        name=(sec.get('section_name') or '').strip(); paras=sec.get('paragraphs',[]) or []
        parts.append(('## '+name+'\n' if name else '')+'\n\n'.join(paras))
    return '\n\n'.join(x for x in parts if x.strip())

def qasper():
    for fn,split in [('qasper-train-v0.3.json','train'),('qasper-dev-v0.3.json','validation'),('qasper-test-v0.3.json','test')]:
        data=json.load((RAW/'qasper'/fn).open(encoding='utf-8')); rows=[]
        for pid,paper in data.items():
            context=paper_context(paper)
            for qa in paper.get('qas',[]):
                answers=[]; evidence=[]; answer_types=[]; unanswerable=[]
                for ann in qa.get('answers',[]):
                    a=ann.get('answer',{}); unanswerable.append(bool(a.get('unanswerable',False)))
                    if a.get('free_form_answer'): answers.append(a['free_form_answer']) ; answer_types.append('free_form')
                    answers.extend(a.get('extractive_spans') or []); answer_types.extend(['extractive']*len(a.get('extractive_spans') or []))
                    if a.get('yes_no') is not None: answers.append('yes' if a['yes_no'] else 'no'); answer_types.append('yes_no')
                    evidence.extend(a.get('evidence') or []); evidence.extend(a.get('highlighted_evidence') or [])
                dedup=[]
                for x in answers:
                    if x and x not in dedup: dedup.append(x)
                ev=[]
                for x in evidence:
                    if x and x not in ev: ev.append(x)
                rows.append({'id':qa.get('question_id') or f'{pid}-{len(rows)}','context':context,'question':qa.get('question',''),'answers':dedup,'answer_starts':[],'evidence':ev,'metadata':{'paper_id':pid,'title':paper.get('title'),'abstract':paper.get('abstract'),'section_names':[s.get('section_name') or '' for s in paper.get('full_text',[])],'figures_and_tables':paper.get('figures_and_tables',[]),'context_type':'full_paper','source_url':f'https://arxiv.org/abs/{pid}' if re.match(r'^\d{4}\.\d+',pid) else None,'nlp_background':qa.get('nlp_background'),'topic_background':qa.get('topic_background'),'paper_read':qa.get('paper_read'),'question_writer':qa.get('question_writer'),'answer_types':sorted(set(answer_types)),'unanswerable':all(unanswerable) if unanswerable else None},'dataset':'qasper','split':split})
        write_rows('qasper',split,rows)

def scidqa():
    full=pickle.load((RAW/'scidqa'/'papers_fulltext_nougat.pkl').open('rb'))
    tabs=pickle.load((RAW/'scidqa'/'relevant_ptabs.pkl').open('rb'))
    for fn,split in [('scidqa.xlsx','train'),('SciDQA_MultiDocQA.xlsx','multidoc')]:
        df=pd.read_excel(RAW/'scidqa'/fn); rows=[]
        for i,r in df.iterrows():
            if pd.isna(r.get('pid')) or pd.isna(r.get('que')) or pd.isna(r.get('ans')): continue
            pid=str(r['pid']); version=str(r.get('version','Initial')); phase='initial' if version.lower().startswith('init') else 'final'; context=full.get(phase,{}).get(pid,'')
            if not context: context=(tabs.get(pid,{}).get('title','')+'\n\n'+tabs.get(pid,{}).get('abs','')).strip()
            meta={'paper_id':pid,'review_id':str(r.get('rid','')),'year':int(r['year']) if pd.notna(r.get('year')) else None,'venue':r.get('venue'),'decision':r.get('decision'),'paper_version':version,'context_type':'full_paper' if full.get(phase,{}).get(pid,'') else 'title_abstract','source_url':r.get('Paper URL') if 'Paper URL' in df.columns and pd.notna(r.get('Paper URL')) else None}
            rows.append({'id':f"scidqa-{r['id']}",'context':context,'question':str(r['que']),'answers':[str(r['ans'])],'answer_starts':[],'evidence':[],'metadata':meta,'dataset':'scidqa','split':split})
        write_rows('scidqa',split,rows)

if __name__=='__main__': qasper(); scidqa()
