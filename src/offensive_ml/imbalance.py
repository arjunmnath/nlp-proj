import os, json, math, random, gc
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import StratifiedKFold
from .data import counts, CLASS_NAMES
from .labse import ImbalanceModel, FocalLoss, cb_weights, build_optimizer, encode, set_top_layers, predict, seed_everything
from .metrics import compute_metrics, save_report


def inv_weights(labels,c,power=1.): return (1/np.maximum(c,1).astype(float)**power)[np.asarray(labels)]

def posthoc_predict(probs,prior,tau): return (np.log(np.clip(probs,1e-12,None))-tau*np.log(np.clip(prior,1e-12,None))).argmax(1)

def bootstrap_ci(y,pred,n=1000,seed=0):
    rng=np.random.default_rng(seed); vals=[]; labels=np.unique(y)
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y)); vals.append(compute_metrics(y[idx],pred[idx])['macro_f1'])
    return np.percentile(vals,[2.5,97.5])

def run_experiment(name,loss_kind,use_wrs,train_df,val_df,out_dir,cfg,seed=42,wrs_power=1.):
    run_dir=Path(out_dir)/f'{name}_s{seed}'; run_dir.mkdir(parents=True,exist_ok=True)
    done=run_dir/'done.json'
    if done.exists(): return json.loads(done.read_text())
    seed_everything(seed,torch.cuda.is_available())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained('setu4993/LaBSE')
    tr_ids,tr_mask,tr_y=encode(train_df,tok,cfg['max_len']); va_ids,va_mask,va_y=encode(val_df,tok,cfg['max_len'])
    c=counts(tr_y.numpy()); cbw=cb_weights(c,cfg['beta'])
    crit={'ce':torch.nn.CrossEntropyLoss(),'focal':FocalLoss(None,cfg['gamma']),'cb_focal':FocalLoss(cbw,cfg['gamma'])}[loss_kind].to(device)
    ds=TensorDataset(tr_ids,tr_mask,tr_y); gen=torch.Generator().manual_seed(seed)
    if use_wrs:
        sw=torch.as_tensor(inv_weights(tr_y.numpy(),c,wrs_power),dtype=torch.double)
        loader=DataLoader(ds,batch_size=cfg['batch_size'],sampler=WeightedRandomSampler(sw,len(ds),replacement=True,generator=gen))
    else: loader=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,generator=gen)
    model=ImbalanceModel().to(device); opt=build_optimizer(model,cfg['head_lr'],cfg['enc_lr'],cfg['llrd'],cfg['weight_decay'])
    total=len(loader)*cfg['epochs']; sch=__import__('transformers').get_linear_schedule_with_warmup(opt,int(.1*total),total)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=-1; best_ep=0; history=[]
    for ep in range(cfg['epochs']):
        if ep in cfg['unfreeze_schedule']: set_top_layers(model,cfg['unfreeze_schedule'][ep])
        model.train(); loss_sum=0
        for ids,mask,y in loader:
            L=int(mask.sum(1).max()); ids=ids[:,:L].to(device); mask=mask[:,:L].to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'): logits=model(ids,mask); loss=crit(logits.float(),y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(opt); scaler.update(); sch.step(); loss_sum+=loss.item()
        probs=predict(model,va_ids,va_mask,device,cfg['eval_batch_size']); m=compute_metrics(va_y.numpy(),probs.argmax(1)); history.append({'epoch':ep+1,'train_loss':loss_sum/len(loader),**m})
        np.save(run_dir/f'val_probs_ep{ep+1}.npy',probs)
        if m['macro_f1']>best:
            best=m['macro_f1']; best_ep=ep+1; np.save(run_dir/'best_val_probs.npy',probs); torch.save(model.state_dict(),run_dir/'best_model.pt')
    y=va_y.numpy(); probs=np.load(run_dir/'best_val_probs.npy'); pred=probs.argmax(1); np.save(run_dir/'y_val.npy',y); save_report(run_dir/'report.txt',y,pred)
    res={'name':name,'seed':seed,'best_epoch':best_ep,**compute_metrics(y,pred),'ci_raw':bootstrap_ci(y,pred).tolist()}
    (run_dir/'history.json').write_text(json.dumps(history,indent=2)); done.write_text(json.dumps(res,indent=2)); return res

def _script_fraction(text,lo=0x0d00,hi=0x0d7f):
    s=[c for c in text if not c.isspace()]; return sum(lo<=ord(c)<=hi for c in s)/max(1,len(s))

def translit_native(text):
    from indic_transliteration import sanscript
    s=sanscript.transliterate(text,sanscript.MALAYALAM,sanscript.ITRANS)
    for a,b in [('RRi','ru'),('RRI','ru'),('~N','ng'),('~n','nj'),('.N','m'),('.n','m'),('A','aa'),('I','ee'),('U','oo'),('E','e'),('O','o'),('M','m'),('H','h')]: s=s.replace(a,b)
    from .data import clean_text
    return clean_text(s)

def build_augmentation(train_df,out_dir,cfg,seed=42):
    """Reproduces the planning/translation structure of the original R4 augmentation stage."""
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); final=out/'train_augmented.csv'
    if final.exists(): return pd.read_csv(final,keep_default_na=False)
    c=counts(train_df.label); target=int(math.ceil(cfg['aug_ratio']*c[0])); need={k:max(0,target-int(c[k])) for k in range(1,5)}
    rng=random.Random(seed); plans=[]
    for cls in range(1,5):
        idx=list(train_df.index[train_df.label==cls]); rng.shuffle(idx); planned=0
        for r in range(cfg['aug_max_per_sample']):
            for i in idx:
                if planned>=need[cls]: break
                if _script_fraction(train_df.at[i,'text'])>.3:
                    method='translit' if r==0 else ['bt_en','bt_hi','bt_ta'][min(r-1,2)]
                    plans.append((i,cls,method)); planned+=1
    plan=pd.DataFrame(plans,columns=['src_idx','label','method']); plan.to_csv(out/'plan.csv',index=False)
    # Translation generation is intentionally explicit and cached. This keeps expensive NLLB calls out of imports.
    synth=[]
    if len(plan):
        tok=AutoTokenizer.from_pretrained(cfg['nllb_model']); model=AutoModelForSeq2SeqLM.from_pretrained(cfg['nllb_model'],torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu').eval(); device=next(model.parameters()).device
        piv={'bt_en':'eng_Latn','bt_hi':'hin_Deva','bt_ta':'tam_Taml'}
        from tqdm import tqdm
        @torch.no_grad()
        def trans(texts,src,tgt,sample):
            tok.src_lang=src; outs=[]
            for i in range(0,len(texts),64):
                enc=tok(texts[i:i+64],return_tensors='pt',padding=True,truncation=True,max_length=96).to(device)
                kw={'do_sample':True,'top_p':.9,'temperature':.8} if sample else {'do_sample':False,'num_beams':1}
                gen=model.generate(**enc,forced_bos_token_id=tok.convert_tokens_to_ids(tgt),max_new_tokens=96,**kw); outs += tok.batch_decode(gen,skip_special_tokens=True)
            return outs
        for method,g in plan.groupby('method'):
            src=[train_df.at[i,'text'] for i in g.src_idx]
            if method=='translit': outs=[translit_native(t) for t in src]
            else:
                pivots=trans(src,'mal_Mlym',piv[method],False); outs=trans(pivots,piv[method],'mal_Mlym',True)
            for i,y,t in zip(g.src_idx,g.label,outs): synth.append((translit_native(t) if method=='translit' else t,int(y),method,i))
        del model,tok; gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    s=pd.DataFrame(synth,columns=['text','label','method','src_idx']) if synth else pd.DataFrame(columns=['text','label','method','src_idx'])
    if len(s): s=s[(s.text.str.len()>=3)&(~s.text.isin(set(train_df.text)))].drop_duplicates('text')
    base=train_df.assign(method='orig',src_idx=-1); result=pd.concat([base,s],ignore_index=True); result.to_csv(final,index=False); return result
