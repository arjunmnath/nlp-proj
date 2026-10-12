import argparse, json
from pathlib import Path
import pandas as pd
from offensive_ml.data import read_fire
from offensive_ml.imbalance import run_experiment, build_augmentation
from offensive_ml.config import IMBALANCE

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-path',required=True); ap.add_argument('--dev-path',required=True); ap.add_argument('--output-dir',default='outputs/imbalance'); ap.add_argument('--with-augmentation',action='store_true'); args=ap.parse_args()
    tr,_=read_fire(args.train_path); va,_=read_fire(args.dev_path)
    tr=tr.drop_duplicates(['text','label']).reset_index(drop=True); overlap=tr.text.isin(set(va.text)); tr=tr[~overlap].reset_index(drop=True)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    configs=[('R0_ce','ce',False,1.),('R1_focal_wrs','focal',True,1.),('R2_cbfocal','cb_focal',False,1.),('R3_cbfocal_wrs','cb_focal',True,1.)]
    results=[]
    for n,l,w,p in configs: results.append(run_experiment(n,l,w,tr,va,out,IMBALANCE,42,p))
    if args.with_augmentation:
        aug=build_augmentation(tr,out/'augmentation',IMBALANCE,42)
        results.append(run_experiment('R4_cbfocal_wrs_aug','cb_focal',True,aug,va,out,IMBALANCE,42,1.))
        results.append(run_experiment('R4b_cbfocal_aug','cb_focal',False,aug,va,out,IMBALANCE,42,1.))
    results.append(run_experiment('R5_cbfocal_softwrs','cb_focal',True,tr,va,out,IMBALANCE,42,.5))
    (out/'summary.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__': main()
