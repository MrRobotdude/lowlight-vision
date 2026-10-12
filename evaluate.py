from pathlib import Path
import os,json
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
p=argparse.ArgumentParser(description='Evaluate a saved model against prepared validation images.')
p.add_argument('--data',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('results'))
args=p.parse_args()
root=args.data
out=args.output;out.mkdir(parents=True,exist_ok=True)
model=tf.keras.models.load_model(args.model,compile=False,safe_mode=True)
names=['Boat','Bicycle','Bus','Car','Motorbike']
paths=[];actual=[]
for i,name in enumerate(names):
 for p in sorted((root/name).iterdir()):
  if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']:
   paths.append(p);actual.append(i)
predictions=[]
for start in range(0,len(paths),16):
 batch=[cv2.cvtColor(cv2.resize(cv2.imread(str(p)),(224,224)),cv2.COLOR_BGR2RGB)/255.0 for p in paths[start:start+16]]
 predictions.extend(model(np.array(batch,dtype=np.float32),training=False).numpy().tolist())
 print(f'{min(start+16,len(paths))}/{len(paths)}',flush=True)
probs=np.array(predictions);pred=probs.argmax(1);actual=np.array(actual)
cm=np.zeros((5,5),dtype=int)
for a,b in zip(actual,pred):cm[a,b]+=1
metrics={'samples':len(paths),'accuracy':float((actual==pred).mean()),'classes':names,'confusion_matrix':cm.tolist(),'scope':'Existing validation split reused during model selection; not an independent test or production benchmark.','class_results':[]}
for i,name in enumerate(names):
 tp=int(cm[i,i]);support=int(cm[i].sum());precision=tp/max(1,int(cm[:,i].sum()));recall=tp/max(1,support)
 metrics['class_results'].append({'class':name,'support':support,'precision':precision,'recall':recall})
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(cm,cmap='Blues');ax.set_xticks(range(5),names);ax.set_yticks(range(5),names);ax.set_xlabel('Predicted category');ax.set_ylabel('Actual category');ax.set_title(f'Low-light image classification · {len(paths)} validation images\nAccuracy: {metrics["accuracy"]:.1%} · existing model-selection split',pad=20)
for i in range(5):
 for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#18334a')
fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'confusion-matrix.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5));x=np.arange(5);ax.bar(x-.18,[r['precision'] for r in metrics['class_results']],.36,label='Precision',color='#176e8a');ax.bar(x+.18,[r['recall'] for r in metrics['class_results']],.36,label='Recall',color='#d3985a');ax.set_xticks(x,names);ax.set_ylim(0,1.1);ax.set_ylabel('Score');ax.set_title('Where the classifier succeeds — and where it struggles');ax.legend();ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(out/'class-results.png',dpi=180);plt.close(fig)
print(json.dumps(metrics,indent=2))
