import torch, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler,TensorDataset,ConcatDataset
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score,accuracy_score,classification_report
from .labse import seed_everything

LABEL_DICT={'Not_offensive':0,'Offensive_Targeted_Insult_Group':3,'Offensive_Targeted_Insult_Individual':2,'Offensive_Targeted_Insult_Other':4,'Offensive_Untargetede':1}

def tokenize(sentences,max_len=256):
    tok=AutoTokenizer.from_pretrained('setu4993/LaBSE',use_fast=False); ids=[]; masks=[]
    for s in sentences:
        e=tok.encode_plus(s,add_special_tokens=True,max_length=max_len,padding='max_length',return_attention_mask=True,return_tensors='pt',truncation=True); ids.append(e['input_ids']); masks.append(e['attention_mask'])
    return torch.cat(ids),torch.cat(masks)

def load_mal(path):
    import pandas as pd
    df=pd.read_csv(path); return df

def load_olid(path):
    rows=[]
    with open(path,encoding='utf8') as f:
        next(f)
        for line in f:
            p=line.rstrip().split('\t');
            if len(p)<3: continue
            lab=p[-1]
            if lab not in LABEL_DICT: continue
            rows.append((' '.join(p[1:-1]).replace('#','').lower(),LABEL_DICT[lab]))
    return rows

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.embeddings=AutoModel.from_pretrained('setu4993/LaBSE'); self.dropout=nn.Dropout(.3); self.classifier=nn.Linear(1536,5)
    def forward(self,ids,mask):
        out=self.embeddings(ids,mask).last_hidden_state; return self.classifier(self.dropout(torch.cat((out.mean(1),out.max(1).values),1)))

def run(mal_train,mal_dev,olid_train,out_path,seed=42,epochs=4,lr=3e-5):
    import pandas as pd
    seed_everything(seed,True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mal=pd.read_csv(mal_train); dev=pd.read_csv(mal_dev); olid=load_olid(olid_train)
    def labels(df): return [LABEL_DICT.get(x,0) for x in df['Labels']]
    tr_text=list(mal['Text'])+[x[0] for x in olid]; tr_y=torch.tensor(labels(mal)+[x[1] for x in olid],dtype=torch.long); va_text=list(dev['Text']); va_y=torch.tensor(labels(dev),dtype=torch.long)
    ti,tm=tokenize(tr_text); vi,vm=tokenize(va_text); train=TensorDataset(ti,tm,tr_y); val=TensorDataset(vi,vm,va_y)
    loader=DataLoader(train,sampler=RandomSampler(train),batch_size=32); vloader=DataLoader(val,sampler=SequentialSampler(val),batch_size=128)
    counts=np.bincount(tr_y.numpy(),minlength=5); weights=torch.tensor([1+np.log(len(train)/max(1,c)) for c in counts],dtype=torch.float32).to(device)
    model=Model().to(device); opt=AdamW(model.parameters(),lr=lr,eps=1e-8); sch=get_linear_schedule_with_warmup(opt,0,len(loader)*epochs); crit=nn.CrossEntropyLoss(weight=weights); scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda'); best=-1
    for ep in range(epochs):
        model.train()
        for ids,mask,y in loader:
            ids,mask,y=ids.to(device),mask.to(device),y.to(device); opt.zero_grad()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'): loss=crit(model(ids,mask),y)
            scaler.scale(loss).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(opt); scaler.update(); sch.step()
        model.eval(); pred=[]
        with torch.no_grad():
            for ids,mask,_ in vloader: pred.extend(model(ids.to(device),mask.to(device)).argmax(1).cpu().numpy())
        f=f1_score(va_y.numpy(),pred,average='weighted'); print(f'Epoch {ep+1}: weighted F1={f:.4f}')
        if f>best: best=f; torch.save({'model_state_dict':model.state_dict(),'epoch':ep+1,'weighted_f1':f},out_path)
    report=classification_report(va_y.numpy(),pred,target_names=['NOT','UNT','IND','GRP','OTH'],zero_division=0); open(str(out_path)+'.txt','w').write(report); print(report); return best
