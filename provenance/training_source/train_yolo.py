from __future__ import annotations
import json, shutil
from pathlib import Path
import torch
from ultralytics import YOLO

root=Path('/root/autodl-tmp/handwash_self_scene_refresh_20261019')
data=root/'imagefolder'; run=root/'runs'/'yolo_1080'; models=root/'models'; models.mkdir(exist_ok=True)
if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
torch.manual_seed(20261019)
# YAML architecture creates a new model with random initialization; no .pt checkpoint is loaded.
for batch_size in (4,2,1):
    run=root/'runs'/f'yolo_1080_b{batch_size}_retry1'
    if (run/'weights/best.pt').exists(): raise FileExistsError(run/'weights/best.pt')
    model=YOLO('yolo26m-cls.yaml')
    try:
        model.train(data=str(data),imgsz=1080,epochs=60,patience=10,batch=batch_size,device=0,workers=6,
                    amp=True,cache=False,project=str(root/'runs'),name=f'yolo_1080_b{batch_size}_retry1',exist_ok=False,
                    seed=20261019,pretrained=False,plots=True,verbose=True)
        break
    except torch.cuda.OutOfMemoryError:
        if batch_size==1: raise
        torch.cuda.empty_cache()
        print(f'CUDA OOM with batch={batch_size}; retrying from random initialization at lower batch size',flush=True)
best=run/'weights/best.pt'
if not best.is_file(): raise RuntimeError(f'Missing best checkpoint {best}')
shutil.copy2(best,models/'yolo_best.pt')
trained=YOLO(str(best))
metrics=trained.val(data=str(data),split='test',imgsz=1080,batch=4,device=0,workers=4,plots=True)
payload={'model':'YOLO26m-cls','initialization':'random from yolo26m-cls.yaml','input_size':[1080,1080],'batch_size':batch_size,
         'dataset':'local self-captured scene only','train_video_ids':[1,2,3,5,6,7,9,11,13,15,17,19,21,23],
         'validation_video_ids':[10,12,14,16,18,20,22,24],'test_video_ids':[4,8],
         'best_checkpoint':str(models/'yolo_best.pt'),'best_epoch':int(getattr(metrics,'epoch',-1)),
         'test_top1':float(metrics.top1),'test_top5':float(metrics.top5),
         'test_results_dict':{k:float(v) for k,v in metrics.results_dict.items() if isinstance(v,(int,float))}}
(models/'yolo_metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print('YOLO_DONE',json.dumps(payload,ensure_ascii=False),flush=True)
