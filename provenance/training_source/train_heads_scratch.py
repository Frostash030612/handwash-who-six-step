import os,sys,csv,pathlib,json,random
os.environ['YOLO_CONFIG_DIR']='/root/autodl-tmp/handwash_six_only_scratch_20261018/cache'
sys.path.insert(0,'/root/autodl-tmp/handwash/src')
import cv2,numpy as np,torch
from ultralytics import YOLO
from handwash.pipelines.exp_demo import EXP_CLASS_NAMES,ExpFeatureExtractor,ExpTemporalHead,file_sha256
from handwash.pipelines.boundary_gru import BoundaryAwareGRU,boundary_targets_from_consensus,weak_boundary_intervals_from_consensus,weak_interval_loss,decode_boundaries
from sklearn.metrics import accuracy_score,f1_score,classification_report
root=pathlib.Path('/root/autodl-tmp/handwash_self_scene_refresh_20261019');device='cuda:0';torch.set_num_threads(4);random.seed(20261018);np.random.seed(20261018);torch.manual_seed(20261018);torch.cuda.manual_seed_all(20261018)
yolo_path=root/'models/yolo_best.pt'
ext=ExpFeatureExtractor(yolo_path,device=device,cache_dir=root/'cache')
data={}
for v in range(1,25):
 d=root/'frames'/f'{v}_frames'
 with (d/'frame_labels.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 ims=[cv2.imread(str(root/'prepared'/f'{v}_frames'/r['image'])) for r in rows];ys=[]
 for r in rows:
  c=r.get('movement_code_train',r.get('movement_code',''))
  ys.append(int(c) if r.get('valid_for_training','').lower()=='true' and c in '0123456' else -100)
 fs=[]
 for i in range(0,len(ims),2):
  if any(x is None for x in ims[i:i+32]): raise RuntimeError(f'bad image v{v}')
  fs.append(ext.forward_bgr(ims[i:i+2])[0])
 data[v]=(np.concatenate(fs),np.array(ys,dtype=np.int64))
print('features',ext.feature_dim,flush=True)
fitids=[1,2,3,5,6,7,9,11,13,15,17,19,21,23];valids=[10,12,14,16,18,20,22,24];trainids=[1,2,3,5,6,7,*range(9,25)];testids=[4,8]
def stats(ids):
 xx=np.concatenate([data[v][0][data[v][1]>=0] for v in ids])
 return torch.tensor(xx.mean(0),device=device),torch.tensor(xx.std(0).clip(1e-4),device=device)
def pack(v):
 x,y=data[v];t,m=boundary_targets_from_consensus(np.where(y>=0,y,-1)); intervals=weak_boundary_intervals_from_consensus(np.where(y>=0,y,-1),max_gap=3)
 return torch.tensor(x,device=device)[None],torch.tensor(y,device=device)[None],torch.tensor(t,device=device),torch.tensor(m,device=device),[(0,a,z) for a,z in intervals]
def make(cls,ids):
 model=cls(feature_dim=ext.feature_dim,hidden=128,num_classes=7,dropout=.3).to(device)
 model.feature_mean.copy_(stats(ids)[0]);model.feature_std.copy_(stats(ids)[1]);return model
def preds(model,ids,boundary=False,threshold=.25,gap=3):
 model.eval(); yy=[];pp=[]
 with torch.no_grad():
  for v in ids:
   x,y,*_=pack(v)
   out=model(x)
   if boundary:
    logits,edge,_=out
    probs=torch.softmax(logits[0],-1).cpu().numpy(); edges=torch.sigmoid(edge[0]).cpu().numpy()
    p=decode_boundaries(probs,edges,threshold=threshold,min_gap=gap).argmax(-1)
   else:p=out[0].argmax(-1)[0].cpu().numpy()
   y=data[v][1];ok=y>=0;yy.extend(y[ok]);pp.extend(p[ok])
 return np.asarray(yy),np.asarray(pp)
def scores(model,ids,boundary=False,threshold=.25,gap=3):
 y,p=preds(model,ids,boundary,threshold,gap)
 return float(f1_score(y,p,average='macro',zero_division=0)),float(accuracy_score(y,p))
def fit_one(model,boundary=False,epochs=50,ids=fitids,valid=valids):
 opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
 best=(-1,1,None,None);stale=0
 for ep in range(1,epochs+1):
  model.train()
  for v in ids:
   x,y,t,mask,ints=pack(v);opt.zero_grad()
   out=model(x)
   if boundary:
    logits,edge,_=out
   else:logits,_=out
   loss=torch.nn.functional.cross_entropy(logits.reshape(-1,7),y.reshape(-1),ignore_index=-100)
   if boundary:
    if mask.any():loss=loss+0.35*torch.nn.functional.binary_cross_entropy_with_logits(edge[0][mask],t[mask])
    loss=loss+0.5*weak_interval_loss(edge,ints)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  if boundary:
   options=[]
   for th in [.1,.15,.2,.25,.3,.35,.4,.5]:
    for gap in [2,3,4]:
     f,a=scores(model,valid,True,th,gap);options.append((f,a,th,gap))
   current=max(options)
  else:
   f,a=scores(model,valid);current=(f,a,None,None)
  if current[0]>best[0]:
   best=(current[0],ep,current[2],current[3]);stale=0
  else:stale+=1
  if ep%5==0 or ep==1:print('val',boundary,ep,'macro/acc/th/gap',current,flush=True)
  if stale>=10:break
 return best
out=root/'models';out.mkdir(exist_ok=True)
# Phase A: choose epochs and boundary decoder only from videos 2 and 7.
g=make(ExpTemporalHead,fitids);gbest=fit_one(g,False)
b=make(BoundaryAwareGRU,fitids);bbest=fit_one(b,True)
print('selected',gbest,bbest,flush=True)
# Phase B: independent random initialization again; refit on all six training videos.
def final_fit(cls,boundary,best):
 model=make(cls,trainids);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
 for ep in range(best[1]):
  model.train()
  for v in trainids:
   x,y,t,mask,ints=pack(v);opt.zero_grad();outp=model(x)
   if boundary: logits,edge,_=outp
   else: logits,_=outp
   loss=torch.nn.functional.cross_entropy(logits.reshape(-1,7),y.reshape(-1),ignore_index=-100)
   if boundary:
    if mask.any():loss=loss+0.35*torch.nn.functional.binary_cross_entropy_with_logits(edge[0][mask],t[mask])
    loss=loss+0.5*weak_interval_loss(edge,ints)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  if ep%5==0:print('final',boundary,ep+1,float(loss.detach()),flush=True)
 return model
g=final_fit(ExpTemporalHead,False,gbest)
b=final_fit(BoundaryAwareGRU,True,bbest)
metrics={}
for key,model,is_b,best in [('yolo_gru',g,False,gbest),('boundary_gru',b,True,bbest)]:
 threshold=best[2] if is_b else .25;gap=best[3] if is_b else 3
 yy,pp=preds(model,testids,is_b,threshold,gap)
 metrics[key]={'accuracy':float(accuracy_score(yy,pp)),'macro_f1':float(f1_score(yy,pp,average='macro',zero_division=0)),'support':int(len(yy)),'internal_validation':{'macro_f1':best[0],'selected_epoch':best[1]},'decoder':{'threshold':threshold,'min_gap':gap} if is_b else None,'report':classification_report(yy,pp,labels=list(range(7)),target_names=list(EXP_CLASS_NAMES),output_dict=True,zero_division=0)}
 torch.save({'format':'sixonly-'+key+'/1','class_names':list(EXP_CLASS_NAMES),'exp_sha256':file_sha256(yolo_path),'fps':5.0,'image_size':1080,'initialization':'random temporal head; local-scene YOLO backbone only','hparams':model.hparams,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'metrics':metrics[key]},out/f'{key}.pt')
(root/'head_metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps({k:{'accuracy':v['accuracy'],'macro_f1':v['macro_f1']} for k,v in metrics.items()}),flush=True)
