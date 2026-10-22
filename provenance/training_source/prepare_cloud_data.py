from __future__ import annotations
import json, shutil, tarfile
from pathlib import Path
import cv2

root=Path('/root/autodl-tmp/handwash_self_scene_refresh_20261019')
archive=root/'local_scene_frames.tar'
manifest=json.loads((root/'dataset_manifest.json').read_text())
with tarfile.open(archive,'r') as tf: tf.extractall(root, filter='data')
classes=manifest['class_names']
shutil.rmtree(root/'imagefolder',ignore_errors=True)
for split in ('train','val','test'):
    for cls in classes: (root/'imagefolder'/split/cls).mkdir(parents=True,exist_ok=True)

for clip in manifest['clips']:
    cid=clip['id']; folder=root/'frames'/f'{cid}_frames'
    rows=list(__import__('csv').DictReader((folder/'frame_labels.csv').open(encoding='utf-8-sig',newline='')))
    target_dir=root/'prepared'/f'{cid}_frames';target_dir.mkdir(parents=True,exist_ok=True)
    for row in rows:
        src=folder/row['image']; dst=target_dir/row['image']
        im=cv2.imread(str(src),cv2.IMREAD_COLOR)
        if im is None: raise RuntimeError(f'Cannot decode {src}')
        h,w=im.shape[:2]; scale=1080/max(h,w)
        nw,nh=round(w*scale),round(h*scale)
        resized=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
        top=(1080-nh)//2; bottom=1080-nh-top; left=(1080-nw)//2; right=1080-nw-left
        square=cv2.copyMakeBorder(resized,top,bottom,left,right,cv2.BORDER_REPLICATE)
        if square.shape[:2]!=(1080,1080): raise RuntimeError(f'Unexpected size {square.shape}: {src}')
        if not cv2.imwrite(str(dst),square,[cv2.IMWRITE_JPEG_QUALITY,92]): raise RuntimeError(f'Cannot write {dst}')
        code=row.get('movement_code_train') or row.get('movement_code') or ''
        valid=row.get('valid_for_training','').lower()=='true' and code in '0123456'
        if valid:
            cls=classes[int(code)]; split='train' if clip['split']=='fit' else clip['split']
            out=root/'imagefolder'/split/cls/f'{cid:02d}_{row["image"]}'
            if not out.exists(): out.symlink_to(dst)
    print(f"prepared {cid:02d} {clip['name']} {len(rows)} frames -> {clip['split']}",flush=True)
summary={**manifest,'preprocessing':{'square_px':1080,'aspect_ratio':'preserved by fit+replicate-pad','jpg_quality':92},
         'training':'self-captured scene only','initialization':'random; YAML architecture, no checkpoint'}
(root/'prepared_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print('READY',json.dumps({'clips':len(manifest['clips']),'frames':manifest['total_frames'],'valid':manifest['total_valid_frames'],'imagefolder':str(root/'imagefolder')},ensure_ascii=False),flush=True)
