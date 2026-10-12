import random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

MODEL_PATH='setu4993/LaBSE'

def seed_everything(seed,use_cuda=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

class ImbalanceModel(nn.Module):
    def __init__(self,model_path=MODEL_PATH):
        super().__init__(); self.encoder=AutoModel.from_pretrained(model_path); self.dropout=nn.Dropout(.3); self.out_dim=self.encoder.config.hidden_size*3
        self.fc1=nn.Linear(self.out_dim,256); self.ln1=nn.LayerNorm(256); self.fc2=nn.Linear(256,128); self.ln2=nn.LayerNorm(128); self.fc3=nn.Linear(128,5); self.drop=nn.Dropout(.3)
        for p in self.encoder.parameters(): p.requires_grad=False
    def forward(self,ids,mask):
        out=self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state.float(); m=mask.unsqueeze(-1).float()
        mean=(out*m).sum(1)/m.sum(1).clamp(min=1); mx=out.masked_fill(m==0,-1e4).max(1).values
        x=self.dropout(torch.cat([out[:,0],mean,mx],1)); x=self.drop(F.relu(self.ln1(self.fc1(x)))); x=self.drop(F.relu(self.ln2(self.fc2(x)))); return self.fc3(x)

def encode(df,tokenizer,max_len):
    e=tokenizer(df.text.tolist(),padding='max_length',truncation=True,max_length=max_len,return_tensors='pt')
    return e['input_ids'],e['attention_mask'],torch.tensor(df.label.values,dtype=torch.long)

def cb_weights(counts,beta=.999):
    c=np.asarray(counts,dtype=float); w=np.zeros_like(c); ok=c>0; w[ok]=(1-beta)/(1-np.power(beta,c[ok])); w[ok]=w[ok]/w[ok].sum()*ok.sum(); return w

class FocalLoss(nn.Module):
    def __init__(self,alpha=None,gamma=2.):
        super().__init__(); self.gamma=gamma; self.register_buffer('alpha',None if alpha is None else torch.tensor(alpha,dtype=torch.float32))
    def forward(self,logits,target):
        logpt=F.log_softmax(logits.float(),-1).gather(1,target[:,None]).squeeze(1); loss=-((1-logpt.exp())**self.gamma)*logpt
        if self.alpha is not None: loss=loss*self.alpha[target]
        return loss.mean()

def set_top_layers(model,n):
    for p in model.encoder.parameters(): p.requires_grad=False
    layers=model.encoder.encoder.layer; selected=layers if n==-1 or n>=len(layers) else (layers[-n:] if n>0 else [])
    for l in selected:
        for p in l.parameters(): p.requires_grad=True

def build_optimizer(model,head_lr=5e-4,enc_lr=2e-5,llrd=.9,weight_decay=.01):
    groups=[]
    def add(params,lr):
        d=[p for p in params if p.ndim>1]; nd=[p for p in params if p.ndim<=1]
        if d: groups.append({'params':d,'lr':lr,'weight_decay':weight_decay})
        if nd: groups.append({'params':nd,'lr':lr,'weight_decay':0.})
    add(list(model.fc1.parameters())+list(model.fc2.parameters())+list(model.fc3.parameters()),head_lr)
    L=len(model.encoder.encoder.layer)
    for i,l in enumerate(model.encoder.encoder.layer): add(list(l.parameters()),enc_lr*llrd**(L-1-i))
    return AdamW(groups,eps=1e-8)

def predict(model,ids,mask,device,bs=128):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(ids),bs):
            m=mask[i:i+bs]; L=int(m.sum(1).max()); logits=model(ids[i:i+bs,:L].to(device),m[:,:L].to(device)); out.append(torch.softmax(logits.float(),-1).cpu())
    return torch.cat(out).numpy()
