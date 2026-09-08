import json, uuid
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
BASE=Path(__file__).resolve().parent; ART=BASE/'artifacts'; OUT=BASE/'static'/'generated'; OUT.mkdir(parents=True,exist_ok=True)
ALLOWED={'jpg','jpeg','png'}
@keras.utils.register_keras_serializable(package='ChestCT')
class DenseNetInputTransform(layers.Layer):
    def call(self,inputs): return keras.applications.densenet.preprocess_input(inputs)
    def get_config(self): return super().get_config()
CLASS_NAMES=json.load(open(ART/'class_names.json')); META=json.load(open(ART/'model_metadata.json')); IMG_SIZE=tuple(META.get('image_size',[224,224]))
MODEL=keras.models.load_model(ART/'selected_model.keras',custom_objects={'DenseNetInputTransform':DenseNetInputTransform})
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=10*1024*1024
def allowed(name): return '.' in name and name.rsplit('.',1)[1].lower() in ALLOWED
def prep(path):
    with Image.open(path) as im: return np.asarray(im.convert('RGB').resize(IMG_SIZE),dtype=np.float32)
def heatmap(batch,idx):
    target=MODEL.get_layer('gradcam_target'); probe=keras.Model(MODEL.inputs,[target.output,MODEL.output])
    with tf.GradientTape() as tape: fmap,p=probe(batch,training=False); score=p[:,idx]
    grads=tape.gradient(score,fmap); w=tf.reduce_mean(grads,axis=(0,1,2)); h=tf.reduce_sum(fmap[0]*w,axis=-1); h=tf.maximum(h,0); m=tf.reduce_max(h); return tf.where(m>0,h/m,h).numpy()
def save_overlay(image,h,path):
    hi=Image.fromarray(np.uint8(np.clip(h,0,1)*255)).resize(IMG_SIZE,Image.Resampling.BILINEAR); a=np.asarray(hi,dtype=np.float32)/255
    color=plt.get_cmap('turbo')(a)[...,:3]*255; Image.fromarray(np.clip(.58*image+.42*color,0,255).astype(np.uint8)).save(path)
@app.route('/',methods=['GET','POST'])
def index():
    result=None; error=None
    if request.method=='POST':
        f=request.files.get('ct_image')
        if not f or not f.filename: error='Please choose a CT image.'
        elif not allowed(f.filename): error='Only JPG, JPEG and PNG files are accepted.'
        else:
            token=uuid.uuid4().hex[:12]; name=token+'_'+secure_filename(f.filename); path=OUT/name; f.save(path)
            try:
                image=prep(path); probs=MODEL.predict(np.expand_dims(image,0),verbose=0)[0]; idx=int(np.argmax(probs)); hm=heatmap(np.expand_dims(image,0),idx)
                grad_name=token+'_gradcam.png'; save_overlay(image,hm,OUT/grad_name)
                ranked=sorted([{'label':CLASS_NAMES[i].replace('_',' ').title(),'probability':float(probs[i])} for i in range(len(CLASS_NAMES))],key=lambda x:x['probability'],reverse=True)
                result={'predicted_class':CLASS_NAMES[idx].replace('_',' ').title(),'confidence':float(probs[idx]),'probabilities':ranked,'input_image':'generated/'+name,'gradcam_image':'generated/'+grad_name,'model_name':META.get('selected_model','Model')}
            except Exception as exc: error='Prediction failed: '+str(exc)
    return render_template('index.html',result=result,error=error,metadata=META)
if __name__=='__main__': app.run(host='127.0.0.1',port=5000,debug=False)
