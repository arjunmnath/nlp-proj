import re, unicodedata
from collections import Counter
import pandas as pd
import numpy as np

CLASS_NAMES = ['NOT','UNT','IND','GRP','OTH']
LABEL_DICT = {
    'Not_offensive': 0,
    'Offensive_Untargetede': 1,
    'Offensive_Targeted_Insult_Individual': 2,
    'Offensive_Targeted_Insult_Group': 3,
    'Offensive_Targeted_Insult_Other': 4,
}

def norm_label(x):
    return str(x).strip().lower().replace(' ', '_').replace('-', '_')
LABEL_NORM = {norm_label(k): v for k,v in LABEL_DICT.items()}
LABEL_NORM['offensive_untargeted'] = 1

_CHILLU={'ന്‍':'ൻ','ര്‍':'ർ','ല്‍':'ൽ','ള്‍':'ൾ','ണ്‍':'ൺ'}
_URL=re.compile(r'https?://\S+|www\.\S+')
_MENTION=re.compile(r'@\w+')
_REPEAT=re.compile(r'([^\W\d_])\1{2,}')

def clean_text(s):
    try:
        import demoji
    except Exception:
        demoji=None
    s=unicodedata.normalize('NFC',str(s))
    for a,b in _CHILLU.items(): s=s.replace(a,b)
    s=s.replace('‍','').replace('‌','').replace('﻿','')
    s=_URL.sub(' url ',s); s=_MENTION.sub(' @user ',s); s=s.replace('#',' ')
    if demoji is not None:
        try:
            for e,t in demoji.findall(s).items(): s=s.replace(e,' '+t+' ')
        except Exception: pass
    s=s.lower(); s=_REPEAT.sub(r'\1\1',s)
    return ' '.join(s.split())

def read_fire(path):
    path=str(path)
    if path.lower().endswith(('.xlsx','.xls')):
        tab=pd.read_excel(path,header=None,dtype=str).fillna('').astype(str)
    else:
        with open(path,encoding='utf-8-sig',errors='replace') as f:
            head=[next(f,'') for _ in range(50)]
        def score(d):
            c=sorted(h.count(d) for h in head if h.strip())
            return (c[0]>0,c[len(c)//2]) if c else (False,0)
        delim=max(['\t',',',';','|'],key=score)
        tab=pd.read_csv(path,sep=delim,header=None,dtype=str,engine='python',quotechar='"',on_bad_lines='skip',encoding='utf-8-sig',keep_default_na=False)
    mapped=tab.apply(lambda col: col.map(lambda v:norm_label(v) in LABEL_NORM))
    lab_col=int(mapped.sum().idxmax())
    if int(mapped[lab_col].sum())==0: raise ValueError(f'No recognised label column in {path}')
    others=[c for c in tab.columns if c!=lab_col]
    txt_col=max(others,key=lambda c:tab[c].str.len().mean())
    keep=tab[lab_col].map(lambda v:norm_label(v) in LABEL_NORM)
    dropped=Counter(v for v in tab[lab_col] if norm_label(v) not in LABEL_NORM)
    out=pd.DataFrame({'text':tab.loc[keep,txt_col].map(clean_text),'label':tab.loc[keep,lab_col].map(lambda v:LABEL_NORM[norm_label(v)]).astype(int)})
    return out[out.text.str.len()>0].reset_index(drop=True), dropped

def read_olid(path):
    rows=[]
    with open(path,encoding='utf-8',errors='replace') as f:
        for i,line in enumerate(f):
            if i==0: continue
            parts=line.rstrip('\n').split('\t')
            if len(parts)<3: continue
            text=parts[1].replace('#','').lower()
            labels=parts[2:]
            if labels[0]=='NOT': lab=0
            elif len(labels)>1 and labels[1]=='UNT': lab=1
            elif len(labels)>2 and labels[2]=='IND': lab=2
            elif len(labels)>2 and labels[2]=='GRP': lab=3
            else: lab=4
            rows.append((text,lab))
    return pd.DataFrame(rows,columns=['text','label'])

def counts(labels): return np.bincount(np.asarray(labels),minlength=5)
