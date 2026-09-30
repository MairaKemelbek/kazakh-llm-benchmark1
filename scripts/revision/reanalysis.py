"""Automatic re-analysis for the revised manuscript.
Produces: sentiment parse failures (Tables 2, 4), QA format compliance and length (Table 3),
extracted-answer scores (Section 4.5), reference counts (Section 3.1), pairwise exact McNemar
tables with Holm correction (Tables 6, 8, 9 and sensitivity analyses in Section 4.6),
the tier-by-condition GEE (Section 4.4) and the power / minimum-detectable-difference
simulation (Section 3.5).
Run from this folder:  python reanalysis.py   ->  revision_reanalysis.xlsx
"""
from stats import *
from extract import *
import statsmodels.api as sm, statsmodels.formula.api as smf
R={}
s=sa(); ctx=qa('test300'); cb=qa('cb300')
# 1 parse failures & sentiment
rows=[]
for m in M:
    x=s[m]; pf=(x.pred==-1).sum()
    rows.append(dict(model=NM[m],n=300,parse_fail=pf,parse_fail_pct=round(pf/300*100,1),acc_paper_excl_fail=round((x.pred==x.gold)[x.pred!=-1].mean(),4),acc_fail_as_wrong=round((x.pred==x.gold).mean(),4),fail_raw=';'.join(x.raw[x.pred==-1].astype(str).unique())))
R['1_sentiment_parse']=pd.DataFrame(rows)
# 2 QA metrics raw vs extracted, format compliance, length
rows=[]
out={}
for reg,d in [('ctx',ctx),('cb',cb)]:
    for m in M:
        x=d[m]; rs=x.gold.map(refs); ex=x.raw.map(extract)
        L=x.raw.astype(str).str.split().str.len()
        r=dict(regime=reg,model=NM[m],
          format_compliant_pct=round(x.raw.map(compliant).mean()*100,1),
          has_explanation_pct=round(x.raw.astype(str).str.lower().str.contains('түсіндірме').mean()*100,1),
          len_words_median=L.median(),len_words_mean=round(L.mean(),1),len_words_p90=L.quantile(.9),
          EM_paper=round(x.em.mean(),3),Cont_paper=round(x.contain.mean(),3),
          Cont_only_pred_in_gold=int(sum(cont_both(p,g)and not cont_fwd(p,g) for p,g in zip(x.raw,rs))),
          Cont_gold_in_pred=round(np.mean([cont_fwd(p,g) for p,g in zip(x.raw,rs)]),3),
          EM_extracted=round(np.mean([em(p,g) for p,g in zip(ex,rs)]),3),
          F1_extracted=round(np.mean([f1(p,g) for p,g in zip(ex,rs)]),3),
          Cont_extracted=round(np.mean([cont_both(p,g) for p,g in zip(ex,rs)]),3))
        rows.append(r)
        out[(reg,m,'cont')]=x.contain.values
        out[(reg,m,'em_ex')]=np.array([em(p,g) for p,g in zip(ex,rs)])
        out[(reg,m,'cont_fwd')]=np.array([cont_fwd(p,g) for p,g in zip(x.raw,rs)])
        out[(reg,m,'cont_ex')]=np.array([cont_both(p,g) for p,g in zip(ex,rs)])
R['2_QA_format_metrics']=pd.DataFrame(rows)
nref=cb['gpt-4o'].gold.map(lambda g: len(refs(g)))
R['3_refs_per_question']=nref.value_counts().sort_index().rename_axis('n_refs').reset_index(name='n_questions')
# 4 McNemar tables
sent={m:(s[m].pred==s[m].gold).astype(int).values for m in M}
sent_ex={m:(s[m].pred==s[m].gold).values for m in M}
summ=[]
def add(name,o,note):
    t=table(o); R['T_'+name]=t.round(4)
    summ.append(dict(analysis=name,sig_raw=(t.p_raw<.05).sum(),sig_holm=(t.p_holm<.05).sum(),note=note))
add('sent_failwrong',sent,'parse failures counted as incorrect (recommended)')
ok=np.all([s[m].pred!=-1 for m in M],axis=0)
add('sent_exclfail',{m:(s[m].pred==s[m].gold).values[ok].astype(int) for m in M},f'items with any parse failure dropped (n={ok.sum()})')
add('ctx_cont',{m:out[('ctx',m,'cont')] for m in M},'paper outcome')
add('cb_cont',{m:out[('cb',m,'cont')] for m in M},'main outcome (Table 9)')
add('cb_cont_goldinpred',{m:out[('cb',m,'cont_fwd')] for m in M},'sensitivity: only gold⊂prediction')
add('cb_em_extracted',{m:out[('cb',m,'em_ex')] for m in M},'sensitivity: EM after extracting "Жауап:" line')
add('cb_cont_extracted',{m:out[('cb',m,'cont_ex')] for m in M},'sensitivity: Containment on extracted answer')
R['4_significance_summary']=pd.DataFrame(summ)
# 6 interaction tier x regime (GEE, clustered by item)
rows=[]
for prov,(big,small) in {'OpenAI':('gpt-4o','gpt-4o-mini'),'Anthropic':('claude-sonnet','claude-haiku')}.items():
    for reg in ['ctx','cb']:
        for m,tier in [(big,1),(small,0)]:
            for i,y in enumerate(out[(reg,m,'cont')]): rows.append(dict(item=i,provider=prov,large=tier,closed=int(reg=='cb'),y=int(y)))
D=pd.DataFrame(rows)
g=smf.gee('y ~ large*closed + provider',groups='item',data=D,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
ci=g.conf_int()
R['6_interaction_GEE']=pd.DataFrame({'term':g.params.index,'coef_logOR':g.params.round(3).values,'OR':np.exp(g.params).round(2).values,'CI_low':np.exp(ci[0]).round(2).values,'CI_high':np.exp(ci[1]).round(2).values,'p':g.pvalues.round(5).values})
# 7 power (min detectable difference, exact McNemar, 80% power, alpha .05 & Holm-worst .005)
from scipy.stats import binom
def power(n,pd_,delta,alpha):
    pb=(pd_+delta)/2; pc=(pd_-delta)/2
    if pc<0: return 1
    # simulate
    rng=np.random.default_rng(0); k=0
    for _ in range(2000):
        x=rng.multinomial(n,[pb,pc,1-pb-pc]); B,C=x[0],x[1]
        if B+C and binomtest(B,B+C).pvalue<alpha: k+=1
    return k/2000
rows=[]
for pd_ in [0.10,0.20,0.30]:
    for alpha in [0.05,0.005]:
        d=next((d for d in np.arange(0.01,pd_,0.005) if power(300,pd_,d,alpha)>=0.8),None)
        rows.append(dict(discordant_rate=pd_,alpha=alpha,min_detectable_diff_pp=round(d*100,1) if d else None))
R['7_power_MDD']=pd.DataFrame(rows)
with pd.ExcelWriter('revision_reanalysis.xlsx') as w:
    for k,t in R.items(): t.to_excel(w,sheet_name=k[:31],index=False)
for k,t in R.items(): print('\n##',k); print(t.to_string(index=False))
