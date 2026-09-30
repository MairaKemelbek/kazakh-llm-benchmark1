from load import *
# Exact McNemar test, Clopper-Pearson CI for the discordance OR, Holm correction, pairwise tables.
from scipy.stats import binomtest, beta
from itertools import combinations
def mcn(a,b):
    a=np.asarray(a).astype(int); b=np.asarray(b).astype(int)
    B=int(((a==1)&(b==0)).sum()); C=int(((a==0)&(b==1)).sum())
    p=binomtest(B,B+C,0.5).pvalue if B+C>0 else 1.0
    # exact CI for OR=B/C via Clopper-Pearson on B/(B+C)
    n=B+C; lo=beta.ppf(.025,B,n-B+1) if B>0 else 0; hi=beta.ppf(.975,B+1,n-B) if B<n else 1
    OR=B/C if C else np.inf
    return B,C,OR,lo/(1-lo) if lo<1 else np.inf,hi/(1-hi) if hi<1 else np.inf,p
def holm(ps):
    ps=np.array(ps); o=np.argsort(ps); adj=np.empty_like(ps); run=0
    for i,k in enumerate(o):
        run=max(run,min(1,(len(ps)-i)*ps[k])); adj[k]=run
    return adj
def table(outcomes):
    rows=[]
    for a,b in combinations(M,2):
        na,nb=outcomes[a].sum(),outcomes[b].sum()
        if nb>na: a,b=b,a; na,nb=nb,na
        B,C,OR,lo,hi,p=mcn(outcomes[a],outcomes[b])
        rows.append(dict(pair=f'{NM[a]} vs {NM[b]}',nA=na,nB=nb,diff=na-nb,b=B,c=C,**{'b-c':B-C},OR=round(OR,2),CI=f'[{lo:.2f}, {hi:.2f}]',p_raw=p))
    t=pd.DataFrame(rows); t['p_holm']=holm(t.p_raw); t=t.sort_values('OR',ascending=False)
    return t
