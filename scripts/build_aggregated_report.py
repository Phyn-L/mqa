#!/usr/bin/env python
import html, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "aggregated"
TOKEN_EST = {"coqa": (2510722,41473943,355), "drop": (1883961,25604998,308), "duorc": (14634440,227872505,1028), "ms_marco": (101700800,102394867,248), "narrativeqa": (115917453,3442374301,73739), "pwc": (8408248,124082218,478), "qasper": (8394293,26526338,5296), "quac": (4484153,50391471,572), "race": (1236845,6993240,457), "scidqa": (17604018,82795131,24215), "squad": (3368646,39362801,160)}
NOTES = {"coqa":"多轮对话；同一 story context 被多个 turn 复用。", "drop":"离散推理；答案不一定是原文 span。", "duorc":"SelfRC 与 ParaphraseRC 是不同设置；含 no_answer。", "ms_marco":"v1.1/v2.1 已合并；需保留 source_version；有大量空答案。", "narrativeqa":"长篇故事/摘要 QA；context 极长，应按 story 隔离。", "pwc":"指令型 context-to-text；包含问答、摘要、抽取、改写和解释。", "qasper":"科研论文 QA；evidence 是关键字段，部分问题无答案。", "quac":"多轮信息寻求对话；问题可能依赖历史 turn。", "race":"四选一阅读理解；答案是选项而非抽取 span。", "scidqa":"科学多文档 QA；multidoc 是 train 子集，含 evidence。", "squad":"抽取式 QA；v1.1/v2.0 context 复用，v2.0 含不可回答问题。"}

def scan(path):
    c = q = empty = noans = evidence = 0; splitc = Counter(); splitq = Counter(); samples = []; maxqa = None; longest = None
    for line in path.open(encoding="utf-8"):
        r = json.loads(line); c += 1; pairs = r.get("qa_pairs", []); q += len(pairs)
        for s in r.get("source_splits", []): splitc[s] += 1
        for p in pairs:
            splitq[p.get("source_split") or "unknown"] += 1; empty += not p.get("answers"); noans += bool((p.get("metadata") or {}).get("no_answer")); evidence += "evidence" in p
        if not samples: samples.append(r)
        if maxqa is None or len(pairs) > len(maxqa.get("qa_pairs", [])): maxqa = r
        if longest is None or len(r.get("context", "")) > len(longest.get("context", "")): longest = r
    samples.extend([x for x in (maxqa, longest) if x is not None and x not in samples])
    return {"contexts":c,"qa":q,"splitc":splitc,"splitq":splitq,"empty":empty,"noans":noans,"evidence":evidence,"samples":samples[:3]}

def fs(n): return f"{n:,}"
def splits(c,q): return "; ".join(f"{s}: {fs(c.get(s,0))} / {fs(q.get(s,0))}" for s in sorted(set(c)|set(q)))
def ctxt(r):
    s=" ".join((r.get("context") or "").split()); return s if len(s)<=600 else s[:600].rstrip()+"…"
def label(r):
    p=(r.get("qa_pairs") or [{}])[0]; m=p.get("metadata") or {}; t=[]
    if m.get("no_answer"): t.append("no_answer")
    if not p.get("answers"): t.append("empty_answer")
    if "evidence" in p: t.append("evidence")
    if p.get("source_version"): t.append("version="+str(p["source_version"]))
    return ", ".join(t) or "普通样例"

def build():
    data=[(p.parent.name,scan(p)) for p in sorted(AGG.glob("*/contexts.jsonl"))]
    tc=sum(x["contexts"] for _,x in data); tq=sum(x["qa"] for _,x in data); tu=sum(TOKEN_EST[d][0] for d,_ in data); te=sum(TOKEN_EST[d][1] for d,_ in data)
    md=["# Aggregated QA 数据集综合报告","","本报告合并数据分布、context token 规模和代表性样例。样例中的 context 仅截断展示，原文保存在 aggregated 目录。","","## 统计口径","","- context：按完整 context 字符串聚合后的记录。","- QA：所有 qa_pairs 的总数。","- token：本地 Qwen3.5 tokenizer 抽样估计；唯一 token 按 context 去重，QA 展开 token 会重复计算。","- split context 相加可能大于总 context，因为同一 context 可跨 split 出现。","","## 总览","","| 数据集 | context | QA | 唯一 context tokens | QA 展开 tokens | 平均 tokens/context | split（context / QA） |","|---|---:|---:|---:|---:|---:|---|"]
    for d,x in data:
        u,e,m=TOKEN_EST[d]; md.append(f"| {d} | {fs(x['contexts'])} | {fs(x['qa'])} | {u/1e6:.2f}M | {e/1e6:.2f}M | {fs(m)} | {splits(x['splitc'],x['splitq'])} |")
    md += [f"| **合计** | **{fs(tc)}** | **{fs(tq)}** | **约 {tu/1e6:.2f}M** | **约 {te/1e9:.2f}B** | — | — |","","## 重点注意事项","","- MS MARCO：v1.1/v2.1 共用 aggregated/ms_marco；跨版本重复 context 3,910 个，涉及 8,341 条 QA。","- SQuAD：v1.1/v2.0 大量复用 context；v2.0 不可回答问题不能删除。","- SciQA：multidoc 是 train 子集，不应相加。","- DuoRC：SelfRC 和 ParaphraseRC 是不同设置；no_answer 需要保留。","- QASPER/SciQA：evidence 是重要评估字段。","- NarrativeQA：context 极长，应按 story 划分。","","## 各数据集特点与代表性样例",""]
    for d,x in data:
        md += [f"### {d}","",f"**特点：** {NOTES[d]}","",f"**分布：** {splits(x['splitc'],x['splitq'])}。",f"**特殊标注：** 空答案 {fs(x['empty'])}；no_answer {fs(x['noans'])}；evidence QA {fs(x['evidence'])}。",""]
        for i,r in enumerate(x["samples"],1):
            p=(r.get("qa_pairs") or [{}])[0]; md += [f"#### 样例 {i}（{label(r)}）","",f"Context（截断）： {ctxt(r)}","",f"Question： {p.get('question','')}",f"Answer： {'；'.join(p.get('answers') or []) or '（空答案）'}",""]
    (ROOT/"AGGREGATED_REPORT.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    esc=html.escape; h=["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Aggregated QA 数据集综合报告</title><style>body{font-family:system-ui,sans-serif;line-height:1.55;max-width:1500px;margin:auto;padding:28px;background:#f6f8fa;color:#17202a}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #bcccdc;padding:8px;text-align:left;vertical-align:top}th{background:#d9eaf7}.note,.card{background:white;border:1px solid #d9e2ec;border-radius:8px;padding:12px;margin:10px 0}.note{border-left:4px solid #f0ad00}.context{background:#f0f4f8;padding:10px;border-radius:5px}</style></head><body><h1>Aggregated QA 数据集综合报告</h1><p>本报告合并数据分布、context token 规模和代表性样例。</p><h2>总览</h2><table><tr><th>数据集</th><th>context</th><th>QA</th><th>唯一 tokens</th><th>QA 展开 tokens</th><th>平均 tokens/context</th><th>split（context / QA）</th></tr>"]
    for d,x in data:
        u,e,m=TOKEN_EST[d]; h.append(f"<tr><td>{esc(d)}</td><td>{fs(x['contexts'])}</td><td>{fs(x['qa'])}</td><td>{u/1e6:.2f}M</td><td>{e/1e6:.2f}M</td><td>{fs(m)}</td><td>{esc(splits(x['splitc'],x['splitq']))}</td></tr>")
    h.append(f"<tr><th>合计</th><th>{fs(tc)}</th><th>{fs(tq)}</th><th>约 {tu/1e6:.2f}M</th><th>约 {te/1e9:.2f}B</th><th>—</th><th>—</th></tr></table><h2>重点注意事项</h2>")
    for n in ["MS MARCO：v1.1/v2.1 共用 aggregated/ms_marco；跨版本重复 context 3,910 个，涉及 8,341 条 QA。","SQuAD：v1.1/v2.0 大量复用 context；v2.0 不可回答问题不能删除。","SciQA：multidoc 是 train 子集，不应相加。","DuoRC：SelfRC 和 ParaphraseRC 是不同设置；no_answer 需要保留。","QASPER/SciQA：evidence 是重要评估字段。","NarrativeQA：context 极长，应按 story 划分。"]: h.append(f"<div class='note'>{esc(n)}</div>")
    h.append("<h2>各数据集特点与代表性样例</h2>")
    for d,x in data:
        h += [f"<h3>{esc(d)}</h3>",f"<p><b>特点：</b>{esc(NOTES[d])}</p>",f"<p>分布：{esc(splits(x['splitc'],x['splitq']))}。特殊标注：空答案 {fs(x['empty'])}；no_answer {fs(x['noans'])}；evidence QA {fs(x['evidence'])}。</p>"]
        for i,r in enumerate(x["samples"],1):
            p=(r.get("qa_pairs") or [{}])[0]; h += [f"<div class='card'><b>样例 {i}（{esc(label(r))}）</b><p>Context（截断）：</p><div class='context'>{esc(ctxt(r))}</div><p><b>Question：</b>{esc(p.get('question',''))}</p><p><b>Answer：</b>{esc('；'.join(p.get('answers') or []) or '（空答案）')}</p></div>"]
    h.append("</body></html>"); (ROOT/"AGGREGATED_REPORT.html").write_text("".join(h),encoding="utf-8")
    print("written",tc,tq)
if __name__=="__main__": build()
