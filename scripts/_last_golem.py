import sys,os,json,numpy as np,time
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
sys.path.insert(0, os.path.dirname(__file__))  # portable
from run_master_all import generate_data, compute_f1, run_golem, save_ckpt, load_ckpt

ckpt=load_ckpt()
p4=ckpt['p4']
key='d=200_K=0.4'
print('Running GOLEM',key)
t0=time.time()
sr=[]
for s in [42,123,456]:
    X,Wt,tr,ep=generate_data(200,200,10,0.4,s)
    Wg=run_golem(X); f1=compute_f1(Wg,Wt,0.3)[0] if Wg is not None else -1
    sr.append({'f1':round(f1,4) if f1>=0 else None})
valid=[s['f1'] for s in sr if s['f1'] is not None]
entry=p4[key]
entry['golem_f1']=round(float(np.mean(valid)),4) if valid else None
entry['golem_best']=round(float(np.max(valid)),4) if valid else None
ckpt['p4']=p4; save_ckpt(ckpt)
print('GOLEM',key,'F1 =',entry.get('golem_f1'),'in',round(time.time()-t0,0),'s')
