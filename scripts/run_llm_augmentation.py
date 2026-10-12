import argparse
from pathlib import Path
import pandas as pd
from offensive_ml.llm_augmentation import finetune_generator,generate,parse_generated,classify
from offensive_ml.config import LLM_AUG

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-path',required=True); ap.add_argument('--dev-path',required=True); ap.add_argument('--output-dir',default='outputs/llm_augmentation'); ap.add_argument('--num-samples',type=int,default=5); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(args.train_path); dev=pd.read_csv(args.dev_path)
    model,tok=finetune_generator(tr,LLM_AUG,out); raw=generate(model,tok,args.num_samples,LLM_AUG); (out/'raw_generation.txt').write_text('\n\n'.join(raw)); texts,labels=parse_generated(raw)
    pd.DataFrame({'Text':texts,'Labels':labels}).to_csv(out/'synthetic_samples.csv',index=False)
    classify(tr,dev,texts,labels,LLM_AUG,out)
if __name__=='__main__': main()
