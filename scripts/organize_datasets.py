#!/usr/bin/env python
"""Normalize local QA corpora while preserving downloaded originals."""
import argparse, csv, glob, hashlib, json, os, zipfile
from pathlib import Path
import pyarrow.parquet as pq
try:
 from bs4 import BeautifulSoup
except ImportError:
 BeautifulSoup = None

ROOT=Path(__file__).resolve().parent.parent; RAW=ROOT/'raw'; OUT=ROOT/'standardized'; OUT.mkdir(exist_ok=True)

def emit(ds, split, rows):
    p=OUT/ds; p.mkdir(exist_ok=True); fn=p/f'{split}.jsonl'
    with fn.open('w',encoding='utf-8') as f:
        seen={}
        for i,r in enumerate(rows):
            r.setdefault('id',f'{ds}-{split}-{i:08d}')
            base=r['id']; seen[base]=seen.get(base,0)+1
            if seen[base]>1: r['id']=f'{base}#dup{seen[base]-1}'
            r.update(dataset=ds,split=split)
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return sum(1 for _ in fn.open(encoding='utf-8'))

def norm_answer(x):
    if x is None:return []
    if isinstance(x,str): return [x]
    return list(x)

def squad():
 for fn in sorted((RAW/'squad').glob('*.json')):
  d=json.load(fn.open()); split=fn.stem.replace('dev','validation').replace('train','train')
  rows=[]
  for a in d['data']:
   for para in a['paragraphs']:
    for q in para['qas']:
     ans=q.get('answers',[]); rows.append({'id':q['id'],'context':para['context'],'question':q['question'],'answers':[x.get('text','') for x in ans],'answer_starts':[x.get('answer_start',-1) for x in ans],'metadata':{'title':a.get('title'),'is_impossible':q.get('is_impossible',False)}})
  emit('squad',split,rows)

def quac():
 for fn in sorted((RAW/'quac').glob('*.json')):
  split='validation' if 'val' in fn.name else 'train'; d=json.load(fn.open()); rows=[]
  for sec in d['data']:
   for dial in sec['paragraphs']:
    for q in dial['qas']:
     aa=q.get('answers',[]); rows.append({'id':q['id'],'context':dial['context'],'question':q['question'],'answers':[x.get('text','') for x in aa],'answer_starts':[x.get('answer_start',-1) for x in aa],'metadata':{'dialogue_id':dial.get('id'),'title':sec.get('title'),'section_title':sec.get('section_title'),'background':sec.get('background'),'followup':q.get('followup'),'yesno':q.get('yesno'),'orig_answer':q.get('orig_answer')}})
  emit('quac',split,rows)

def parquet_file(path, ds, split, transform):
 rows=[]
 for r in pq.read_table(path).to_pylist(): rows.append(transform(r))
 emit(ds,split,rows)

def drop_transform(r):
 a=r.get('answers_spans') or {}; return {'id':r.get('query_id'),'context':r.get('passage',''),'question':r.get('question',''),'answers':a.get('spans',[]),'answer_starts':[],'metadata':{'section_id':r.get('section_id'),'answer_types':a.get('types',[])}}
def coqa_rows(path,split):
 rows=[]
 for r in pq.read_table(path).to_pylist():
  qs=r['questions']['input_text']; ans=r['answers']['input_text']; starts=r['answers']['span_start']
  for i,q in enumerate(qs): rows.append({'id':f"{r['id']}-{i+1}",'context':r['story'],'question':q,'answers':[ans[i]],'answer_starts':[starts[i]],'metadata':{'story_id':r['id'],'source':r.get('source'),'turn_id':r['questions']['turn_id'][i]}})
 emit('coqa',split,rows)

def ms_marco():
 for version in ['v1.1','v2.1']:
  for fn in sorted((RAW/'ms_marco'/'source'/version).glob('*.parquet')):
   split=fn.name.split('-')[0]; rows=[]
   for r in pq.read_table(fn).to_pylist():
    p=r.get('passages') or {}; texts=p.get('passage_text',[]); sel=p.get('is_selected',[]); ctx=[t for t,s in zip(texts,sel) if s] or texts[:1]
    rows.append({'id':f"{version}-{r.get('query_id')}",'context':'\n\n'.join(ctx),'question':r.get('query',''),'answers':norm_answer(r.get('answers')),'answer_starts':[],'metadata':{'version':version,'query_type':r.get('query_type'),'passages':texts,'is_selected':sel,'urls':p.get('url',[])}})
   emit('ms_marco_'+version.replace('.','_'),split,rows)

def race():
 all_rows={}
 for fn in sorted((RAW/'race/data').glob('*/*.txt')):
  split=fn.parent.name; d=json.load(fn.open()); rows=all_rows.setdefault(split,[])
  for i,q in enumerate(d.get('questions',[])):
   ans=d.get('answers',[''])[i] if i<len(d.get('answers',[])) else ''; opts=d.get('options',[[]])[i]
   rows.append({'id':f"{fn.stem}-{i}",'context':d.get('article',''),'question':q,'answers':[opts[ord(ans)-65]] if ans and ord(ans)-65<len(opts) else [],'answer_starts':[],'metadata':{'options':opts,'answer_letter':ans,'source_file':str(fn.relative_to(RAW/'race'))}})
 for split, rows in all_rows.items(): emit('race',split,rows)

def duorc():
 for fn in sorted((RAW/'duorc').glob('*.json')):
  split=fn.stem.rsplit('_',1)[-1].lower(); typ=fn.stem.split('_')[0].lower(); rows=[]
  for item in json.load(fn.open()):
   for q in item.get('qa',[]): rows.append({'id':q.get('id'),'context':item.get('plot',''),'question':q.get('question',''),'answers':q.get('answers',[]),'answer_starts':[],'metadata':{'title':item.get('title'),'no_answer':q.get('no_answer',False),'duorc_type':typ}})
  emit('duorc',typ+'_'+split,rows)

def clean_narrative_story(path):
 """Extract screenplay/story text and remove source-site boilerplate."""
 raw=path.read_text(encoding='utf-8',errors='replace').strip()
 if not raw: return ''
 if raw.lstrip().lower().startswith(('<html','<!doctype')) and BeautifulSoup is not None:
  soup=BeautifulSoup(raw,'html.parser')
  node=soup.find(class_='scrtext') or soup.find('pre')
  if node is None: return ''
  for x in node(['script','style','noscript']): x.decompose()
  text=node.get_text('\n')
 else:
  text=raw
  start=text.find('*** START OF')
  end=text.find('*** END OF')
  if start >= 0: text=text[text.find('\n',start)+1:]
  if end >= 0: text=text[:end]
 lines=[]
 for line in text.splitlines():
  line=line.strip()
  if line: lines.append(line)
 return '\n'.join(lines).strip()

def narrativeqa():
 docs={}
 with (RAW/'narrativeqa/documents.csv').open(encoding='utf-8') as f:
  for r in csv.DictReader(f): docs[r['document_id']]=r
 summaries={}
 if (RAW/'narrativeqa/summaries.csv').exists():
  with (RAW/'narrativeqa/summaries.csv').open(encoding='utf-8') as f:
   for r in csv.DictReader(f): summaries[r['document_id']]=r.get('summary','')
 rows_by={}
 story_cache={}
 with (RAW/'narrativeqa/qaps.csv').open(encoding='utf-8') as f:
  for r in csv.DictReader(f):
   d=docs.get(r['document_id'],{}); ctx=summaries.get(r['document_id'],''); context_type='wikipedia_summary'
   if r['document_id'] in story_cache:
    cached=story_cache[r['document_id']]; ctx,context_type=cached
   else:
    story=RAW/'narrativeqa'/'tmp'/f"{r['document_id']}.content"
    if story.exists() and story.stat().st_size > 19000:
     try:
      text=clean_narrative_story(story)
      if text: ctx=text; context_type='story'
     except OSError: pass
    story_cache[r['document_id']]=(ctx,context_type)
   row={'id':r['document_id']+'-'+hashlib.sha1(r['question'].encode()).hexdigest()[:10],'context':ctx,'question':r['question'],'answers':[r['answer1'],r['answer2']],'answer_starts':[],'metadata':{'document_id':r['document_id'],'set':r['set'],'story_url':d.get('story_url'),'wiki_url':d.get('wiki_url'),'wiki_title':d.get('wiki_title'),'context_type':context_type,'context_note':'Falls back to the official Wikipedia summary when the story download is unavailable.'}}
   rows_by.setdefault(r['set'],[]).append(row)
 for s,rows in rows_by.items(): emit('narrativeqa','validation' if s == 'valid' else s,rows)

def pwc():
 for fn in sorted((RAW/'pwc').glob('PwC_*.jsonl')):
  split=fn.stem.rsplit('_',1)[-1].lower(); rows=[]
  for i,line in enumerate(fn.open(encoding='utf-8')):
   r=json.loads(line)
   rows.append({'id':f'pwc-{split}-{i:08d}','context':r.get('input',''),'question':r.get('prompt',''),'answers':norm_answer(r.get('answer')),'answer_starts':[],'metadata':{'source_file':fn.name}})
  emit('pwc',split,rows)

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--datasets', nargs='+', choices=['squad','quac','drop','coqa','ms_marco','race','duorc','narrativeqa','pwc'], default=['all'])
 selected=set(parser.parse_args().datasets)
 if 'all' in selected: selected={'squad','quac','drop','coqa','ms_marco','race','duorc','narrativeqa','pwc'}
 if 'squad' in selected: squad()
 if 'quac' in selected: quac()
 if 'drop' in selected:
  for fn in sorted((RAW/'drop'/'source').glob('*.parquet')): parquet_file(fn,'drop','train' if 'train' in fn.name else 'validation',drop_transform)
 if 'coqa' in selected: coqa_rows(RAW/'coqa'/'source'/'train.parquet','train'); coqa_rows(RAW/'coqa'/'source'/'validation.parquet','validation')
 if 'ms_marco' in selected: ms_marco()
 if 'race' in selected: race()
 if 'duorc' in selected: duorc()
 if 'narrativeqa' in selected: narrativeqa()
 if 'pwc' in selected: pwc()
 print('standardized output:',sum(1 for _ in OUT.rglob('*.jsonl')),'files')
