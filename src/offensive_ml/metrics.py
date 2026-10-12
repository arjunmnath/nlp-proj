import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

def compute_metrics(y,pred):
    present=np.unique(y)
    pc=f1_score(y,pred,labels=list(range(5)),average=None,zero_division=0)
    return {'macro_f1':float(f1_score(y,pred,labels=present,average='macro',zero_division=0)),
            'macro_f1_all5':float(f1_score(y,pred,labels=list(range(5)),average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),
            'accuracy':float(accuracy_score(y,pred)),
            'f1_per_class':[float(x) for x in pc]}

def save_report(path,y,pred):
    import pandas as pd
    cm=confusion_matrix(y,pred,labels=range(5))
    pd.DataFrame(cm,index=['NOT','UNT','IND','GRP','OTH'],columns=['NOT','UNT','IND','GRP','OTH']).to_csv(str(path).replace('.txt','_confusion.csv'))
    with open(path,'w') as f: f.write(classification_report(y,pred,labels=range(5),target_names=['NOT','UNT','IND','GRP','OTH'],zero_division=0))
