# Human-evaluation scoring (Section 4.7, Tables 10 and 11).
# Inputs are NOT public: the annotation workbooks of annotators A and B, the unblinding key and the
# adjudication file. The script is provided so the procedure can be inspected:
#   python score_human_eval.py annotator_A.xlsx annotator_B.xlsx KEY.xlsx adjudicated.xlsx
# Outputs: inter-annotator agreement (Cohen's kappa, linearly weighted kappa), metric validity
# (FP share, FPR, FNR of EM / Containment variants against strict human labels), human accuracy by
# model, exact McNemar tests on human labels with Holm correction, and sentiment labels vs. stars.
# Final labels: adjudicated value where the annotators disagreed; otherwise the agreed label, or
# Annotator A's label for singly annotated items.
"""
Адам бағалауын өңдеу (v2: A барлық жолды, B ішкі жиынды бағалайды).
  python score_human_eval.py human_eval_annotator_A_v2.xlsx human_eval_annotator_B_v2.xlsx KEY_do_not_share.xlsx [adjudicated.xlsx]
Нәтиже: human_eval_results.xlsx және disagreements.xlsx (A мен B келіспеген жолдар).
Келісім (κ) тек екеуі де бағалаған жолдарда есептеледі; соңғы белгі: A, ал келіспеушілік болса — adjudicated файлдағы мән
(ол болмаса — екеуінің төменгісі, консервативті).
format_ok автоматты: бір жол, «Түсіндірме» жоқ, ≤12 сөз (KEY.format_auto).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score
A,Bf,KEY=sys.argv[1:4]; ADJ=sys.argv[4] if len(sys.argv)>4 else None
QL=['correct','hallucination','ref_problem']
def load(f):
    q=pd.concat([pd.read_excel(f,s) for s in ['CB_closed_book','CX_with_context']])
    q=q[['eval_id']+QL]; s=pd.read_excel(f,'SA_sentiment')[['eval_id','sentiment']]
    return q,s
qa,sa=load(A); qb,sb=load(Bf)
for q in (qa,qb):
    for l in QL: q[l]=q[l].astype(float)
    for l in ['hallucination','ref_problem']: q[l]=q[l].fillna(0)
key=pd.read_excel(KEY,'QA_key'); skey=pd.read_excel(KEY,'Sentiment_key')
qa=qa.merge(key,on='eval_id'); qb=qb.dropna(subset=['correct'])
missing=qa.correct.isna().sum()
O=qa.merge(qb,on='eval_id',suffixes=('','_B'))
OS=sa.merge(sb.dropna(subset=['sentiment']),on='eval_id',suffixes=('','_B'))
out={}
rows=[]
for reg in ['cb','ctx']:
    d=O[(O.regime==reg)&O.correct.notna()]
    for l in QL:
        if len(d)==0: continue
        a,b=d[l].astype(int),d[l+'_B'].astype(int)
        k=cohen_kappa_score(a,b) if len(set(a)|set(b))>1 else np.nan
        rows.append(dict(regime=reg,label=l,n_double=len(d),pct_agree=round((a==b).mean()*100,1),kappa=round(k,3),
          kappa_weighted=round(cohen_kappa_score(a,b,weights='linear'),3) if l=='correct' and len(set(a)|set(b))>1 else None))
if len(OS): rows.append(dict(regime='sentiment',label='sentiment',n_double=len(OS),pct_agree=round((OS.sentiment==OS.sentiment_B).mean()*100,1),kappa=round(cohen_kappa_score(OS.sentiment,OS.sentiment_B),3)))
out['1_agreement']=pd.DataFrame(rows)
# disagreements
dq=O[(O.correct!=O.correct_B)|(O.hallucination!=O.hallucination_B)][['eval_id','correct','correct_B','hallucination','hallucination_B']]
ds=OS[OS.sentiment!=OS.sentiment_B][['eval_id','sentiment','sentiment_B']]
with pd.ExcelWriter('disagreements.xlsx') as w:
    dq.assign(correct_final='',hallucination_final='').to_excel(w,sheet_name='QA',index=False)
    ds.assign(sentiment_final='').to_excel(w,sheet_name='SA',index=False)
# final labels
Q=qa.copy(); S=sa.merge(skey,on='eval_id')
bm=qb.set_index('eval_id')
for l,agg in [('correct',min),('hallucination',max)]:
    d=Q.eval_id.map(bm[l]); dis=d.notna()&(d!=Q[l])
    Q.loc[dis,l]=[agg(x,y) for x,y in zip(Q.loc[dis,l],d[dis])]
sbm=sb.set_index('eval_id').sentiment
if ADJ:
    aq=pd.read_excel(ADJ,'QA').set_index('eval_id'); asd=pd.read_excel(ADJ,'SA').set_index('eval_id')
    for l in ['correct','hallucination']:
        m=aq[l+'_final'].dropna(); Q.loc[Q.eval_id.isin(m.index),l]=Q.eval_id.map(m)
    m=asd.sentiment_final.dropna(); S.loc[S.eval_id.isin(m.index),'sentiment']=S.eval_id.map(m)
Q=Q.dropna(subset=['correct'])
Q['human']=(Q.correct==2).astype(int); Q['human_lenient']=(Q.correct>=1).astype(int)
rows=[]
for reg in ['cb','ctx']:
  for mdl in ['ALL']+sorted(Q.model.unique()):
    d=Q[Q.regime==reg] if mdl=='ALL' else Q[(Q.regime==reg)&(Q.model==mdl)]
    if len(d)==0: continue
    for met in ['EM','Cont','Cont_gold_in_pred','EM_extracted','Cont_extracted']:
        p=d[met]==1; h=d.human==1
        rows.append(dict(regime=reg,model=mdl,metric=met,n=len(d),metric_acc=round(p.mean(),3),human_acc=round(h.mean(),3),
          FP_share_of_metric_correct=round((p&~h).sum()/max(p.sum(),1),3),FPR=round((p&~h).sum()/max((~h).sum(),1),3),
          FNR=round((~p&h).sum()/max(h.sum(),1),3),agreement=round((p==h).mean(),3),
          kappa_vs_human=round(cohen_kappa_score(p,h),3) if p.nunique()>1 and h.nunique()>1 else None))
out['2_metric_validity']=pd.DataFrame(rows)
g=Q.groupby(['regime','model'])
out['3_human_by_model']=pd.DataFrame({'n':g.size(),'human_acc_strict':g.human.mean().round(3),'human_acc_lenient':g.human_lenient.mean().round(3),
  'partial_pct':g.correct.apply(lambda x:(x==1).mean()*100).round(1),'hallucination_pct':g.hallucination.mean().mul(100).round(1),
  'format_ok_pct_auto':g.format_auto.mean().mul(100).round(1),'ref_problem_pct':g.ref_problem.mean().mul(100).round(1)}).reset_index()
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); a=np.empty_like(p); r=0
    for i,k in enumerate(o): r=max(r,min(1,(len(p)-i)*p[k])); a[k]=r
    return a
rows=[]
for reg in ['cb','ctx']:
    W=Q[Q.regime==reg].pivot(index='item',columns='model',values='human').dropna(); res=[]
    for a,b in combinations(W.columns,2):
        if W[b].sum()>W[a].sum(): a,b=b,a
        B=int(((W[a]==1)&(W[b]==0)).sum()); C=int(((W[a]==0)&(W[b]==1)).sum())
        res.append(dict(regime=reg,n_items=len(W),pair=f'{a} vs {b}',nA=int(W[a].sum()),nB=int(W[b].sum()),b=B,c=C,OR=round(B/C,2) if C else np.inf,p_raw=binomtest(B,B+C).pvalue if B+C else 1.0))
    if res: t=pd.DataFrame(res); t['p_holm']=holm(t.p_raw); rows.append(t)
if rows: out['4_mcnemar_human']=pd.concat(rows).round(4)
S['star']=S.star_label.map({1:'pos',0:'neg'})
out['5_sentiment_vs_stars']=pd.crosstab(S.star,S.sentiment.fillna('blank'),margins=True).reset_index()
out['6_notes']=pd.DataFrame({'note':[f'A: unlabeled QA rows = {missing}',f'double-annotated QA rows = {len(O.dropna(subset=["correct","correct_B"]))}, SA = {len(OS)}',
  'human = 1 only if correct == 2 (strict); lenient counts partial (1) as correct',
  'FP_share_of_metric_correct: metric says correct, human says not (R1-1); FNR: human correct, metric says wrong (R1-2)',
  'unadjudicated disagreements: lower correct / higher hallucination used (conservative)']})
with pd.ExcelWriter('human_eval_results.xlsx') as w:
    for k,v in out.items(): v.to_excel(w,sheet_name=k[:31],index=False)
for k,v in out.items(): print('\n##',k); print(v.head(15).to_string(index=False))
