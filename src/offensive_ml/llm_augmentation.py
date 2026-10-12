import ast, json, os
import pandas as pd, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

LABELS=['Not_offensive','Offensive_Untargetede','Offensive_Targeted_Insult_Individual','Offensive_Targeted_Insult_Group','Offensive_Targeted_Insult_Other']

def instruction_following_dataset(df,size=50):
    probs=df['Labels'].map(1/df['Labels'].value_counts())
    sample=df.sample(n=size,weights=probs,random_state=42)
    system='''You are an offensive sentence generator. When the user types "Generate" or "generate", return a JSON object with Text and Label. Label must be one of: Not_offensive, not-malayalam, Offensive_Targeted_Insult_Individual, Offensive_Targeted_Insult_Group, Offensive_Targeted_Insult_Others, Offensive_Untargetede.'''
    out=[]
    for _,r in sample.iterrows():
        out.append({'text':f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nGenerate<|im_end|>\n<|im_start|>assistant\n{{"Text":"{r["Text"]}","Label":"{r["Labels"]}"}}<|im_end|>'})
    return out

def finetune_generator(train_df,cfg,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    model_name=cfg['generator_model']; tokenizer=AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,device_map='auto' if torch.cuda.is_available() else None)
    ds=HFDataset.from_list(instruction_following_dataset(train_df,cfg['few_shot_size']))
    def tok(ex):
        x=tokenizer(ex['text'],max_length=cfg['max_length'],truncation=True,padding='max_length'); x['labels']=x['input_ids'].copy(); return x
    ds=ds.map(tok,batched=True)
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],target_modules=['q_proj','v_proj']))
    args=TrainingArguments(output_dir=str(out_dir/'sarvam-ft-offensive'),per_device_train_batch_size=cfg['per_device_train_batch_size'],gradient_accumulation_steps=cfg['gradient_accumulation_steps'],num_train_epochs=cfg['epochs'],learning_rate=cfg['learning_rate'],logging_steps=10,optim='adamw_torch',save_strategy='no',report_to='none',remove_unused_columns=True,gradient_checkpointing=False,fp16=torch.cuda.is_available())
    Trainer(model=model,args=args,train_dataset=ds,data_collator=DataCollatorForSeq2Seq(tokenizer,pad_to_multiple_of=8,return_tensors='pt')).train()
    return model,tokenizer

def generate(model,tokenizer,n=5,cfg=None):
    prompt='<|im_start|>system\nYou are an offensive sentence generator. Generate an offensive sentence and its corresponding class as a JSON object.<|im_end|>\n<|im_start|>user\nGenerate<|im_end|>\n<|im_start|>assistant\n'
    inputs=tokenizer(prompt,return_tensors='pt').to(model.device); out=[]
    for _ in range(n):
        y=model.generate(**inputs,max_new_tokens=150,do_sample=True,temperature=cfg['generation_temperature'],top_p=cfg['generation_top_p'])
        out.append(tokenizer.decode(y[0][inputs.input_ids.shape[-1]:],skip_special_tokens=True))
    return out

def parse_generated(results):
    texts=[]; labels=[]
    for result in results:
        try:
            s=result.strip().removeprefix('```json').removesuffix('```').strip(); d=ast.literal_eval(s)
            if isinstance(d,dict) and 'Text' in d and 'Label' in d: texts.append(d['Text']); labels.append(d['Label'])
        except Exception: pass
    return texts,labels

class FFClassifier(nn.Module):
    def __init__(self,input_dim,num_classes):
        super().__init__(); self.network=nn.Sequential(nn.Linear(input_dim,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,num_classes))
    def forward(self,x): return self.network(x)

def classify(train_df,dev_df,synthetic_texts,synthetic_labels,cfg,out_dir):
    texts=list(train_df.Text)+synthetic_texts; labels=list(train_df.Labels)+synthetic_labels
    enc=SentenceTransformer(cfg['classifier_encoder']); X=enc.encode(texts,show_progress_bar=True,batch_size=64); Xv=enc.encode(list(dev_df.Text),show_progress_bar=True,batch_size=64)
    le=LabelEncoder(); y=le.fit_transform(labels); yv=le.transform(dev_df.Labels)
    tr=DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(y)),batch_size=cfg['classifier_batch_size'],shuffle=True)
    dv=DataLoader(TensorDataset(torch.tensor(Xv,dtype=torch.float32),torch.tensor(yv)),batch_size=cfg['classifier_batch_size'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=FFClassifier(X.shape[1],len(le.classes_)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg['classifier_lr'],weight_decay=cfg['classifier_weight_decay']); crit=nn.CrossEntropyLoss()
    for _ in range(cfg['classifier_epochs']):
        model.train()
        for x,t in tr: x,t=x.to(device),t.to(device); opt.zero_grad(); loss=crit(model(x),t); loss.backward(); opt.step()
    model.eval(); pred=[]
    with torch.no_grad():
        for x,_ in dv: pred.extend(model(x.to(device)).argmax(1).cpu().numpy())
    report=classification_report(yv,pred,target_names=le.classes_,zero_division=0); print(report); (out_dir/'classification_report.txt').write_text(report); return report
