import argparse
from pathlib import Path
from offensive_ml.combined import run

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mal-train',required=True); ap.add_argument('--mal-dev',required=True); ap.add_argument('--olid-train',required=True); ap.add_argument('--output-dir',default='outputs/combined_olid'); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); run(args.mal_train,args.mal_dev,args.olid_train,out/'best_model.pt',seed=42,epochs=4,lr=3e-5)
if __name__=='__main__': main()
