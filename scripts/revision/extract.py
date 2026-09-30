from load import *
# Normalization, answer extraction (Appendix A, Table A3), format compliance and matching metrics.
import unicodedata
def norm(t):
    t=str(t).lower(); t=''.join(ch for ch in t if not unicodedata.category(ch).startswith('P')); return ' '.join(t.split())
def extract(raw):
    s=str(raw).replace('**','').replace('__','')
    lines=[l.strip().lstrip('#').strip() for l in s.split('\n')]
    lines=[l for l in lines if l]
    for i,l in enumerate(lines):
        m=re.match(r'^жауап\s*:\s*(.*)$',l,flags=re.I)
        if m:
            a=m.group(1).strip()
            if not a and i+1<len(lines): a=lines[i+1]
            return a
    return lines[0] if lines else ''
def compliant(raw):
    s=str(raw).strip(); lines=[l for l in s.split('\n') if l.strip()]
    return int(len(lines)==1 and 'түсіндірме' not in s.lower() and len(s.split())<=12)
def refs(g): return [r for r in str(g).split('|||') if r.strip()]
def em(p,rs): return int(any(norm(p)==norm(r) for r in rs))
def cont_both(p,rs): n=norm(p); return int(bool(n) and any((norm(r) in n) or (n in norm(r)) for r in rs))
def cont_fwd(p,rs): n=norm(p); return int(any(norm(r) and norm(r) in n for r in rs))
def f1(p,rs):
    best=0
    for r in rs:
        a,b=norm(p).split(),norm(r).split()
        c=sum(min(a.count(w),b.count(w)) for w in set(a))
        if c: pr,rc=c/len(a),c/len(b); best=max(best,2*pr*rc/(pr+rc))
    return best
