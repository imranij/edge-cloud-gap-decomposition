import sys; sys.path.insert(0,'src/sens/shortcuts'); sys.path.insert(0,'src/sens'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
import run_shortcuts as R, common as C
from escalation.metrics import balanced_accuracy as ba, threshold_max_ba
res=[]
for name,key in [("threew","a_well_rate"),("skab","a_t_start_tree3"),("hydraulic","a_cycle_1nn")]:
    ds=C.dataset(name); y=ds["y"]; specs={k:(X,mk) for k,X,mk in R.build_specs(name,ds)}
    X,mk=specs[key]; v=[];v5=[];vt=[];pool=[[],[]]
    for sid in C.stored_split_ids(name,"A"):
        sp=C.reconstruct_split(name,"A",sid); f,va,te=sp["fit"],sp["val"],sp["test"]
        if mk=="wellrate": pv=R.well_rate(X[f],y[f],X[va]); pt=R.well_rate(X[f],y[f],X[te])
        else:
            m=mk(); m.fit(X[f],y[f]); pv=m.predict_proba(X[va])[:,1]; pt=m.predict_proba(X[te])[:,1]
        t=threshold_max_ba(pv,y[va]); v.append(ba(y[te],pt>=t)); v5.append(ba(y[te],pt>=.5))
        vt.append(ba(y[te],pt>=threshold_max_ba(pt,y[te])))
        pool[0].append(y[te]); pool[1].append(pt>=t)
    print(name,key,"val-thr %.3f median %.3f | thr0.5 %.3f | test-oracle thr %.3f | pooled %.3f | min %.3f max %.3f"%(np.mean(v),np.median(v),np.mean(v5),np.mean(vt),ba(np.concatenate(pool[0]),np.concatenate(pool[1])),min(v),max(v)))
# hydraulic: 1-NN on order and within-split stats; 3W: in-sample well rate
ds=C.dataset("threew"); y=ds["y"]; w=ds["meta"].well.to_numpy()
p=R.well_rate(w,y,w); print("3W in-sample well rate BA (thr best)", ba(y,p>=threshold_max_ba(p,y)))
ds=C.dataset("hydraulic"); y=ds["y"]; 
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.neighbors import KNeighborsClassifier
for nm,Xh in [("cycle",ds["meta"][["cycle"]].to_numpy(float)),("window_id",np.arange(len(y)).reshape(-1,1).astype(float))]:
    pr=cross_val_predict(KNeighborsClassifier(1),Xh,y,cv=KFold(5,shuffle=True,random_state=0)); print("hyd random 5fold 1NN",nm,ba(y,pr))
print(ds["meta"].cycle.describe())
