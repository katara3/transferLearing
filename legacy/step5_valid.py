import os, sys
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, Activation, MaxPooling2D
import matplotlib.pyplot as plt
import sklearn.metrics
import itertools  
from keras.applications.resnet50 import ResNet50
from keras import optimizers
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from PIL import Image
import yaml
from datetime import datetime

'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
'''

if len(sys.argv) == 1:
    print('설정 파일 경로를 입력 하세요.')
    sys.exit()

print (sys.argv)
config_path = sys.argv[1]
with open(config_path, 'r') as fp:
    config = yaml.safe_load(fp.read())

width = config['width']
height = config['height']
model=load_model(config['model_save_path'])
label_map = config['label_map']
prod = config['prod']
title = 'Product: '+prod.replace('/','')
print('title',title)


t_config=dict()
for config_name in os.listdir(os.path.join(prod,'roi_config')):
    with open(os.path.join(os.path.join(prod,'roi_config'),config_name), 'r') as fp:
        config = yaml.load(fp.read())
    t_sub_config=dict()
    t_sub_config['pos_case'] =  config[list(config.keys())[0]]['pos_case']
    t_sub_config['ng_case'] = config[list(config.keys())[0]]['ng_case_list']
    t_sub_config['x1'] = config[list(config.keys())[0]]['x1']
    t_sub_config['x2'] = config[list(config.keys())[0]]['x2']
    t_sub_config['y1'] = config[list(config.keys())[0]]['y1']
    t_sub_config['y2'] = config[list(config.keys())[0]]['y2']
    t_config[list(config.keys())[0]] = t_sub_config

print(t_config)
y_true=[]
y_pred=[]
def plot_confusion_matrix(cm, title, target_names=None, cmap=None, normalize=True, labels=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        #plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig("valid_"+title+'_'+cur_time+".png",bbox_inces='tight', pad_inches=0, dpi=100)


def pred(img_path,y_true,y_pred,x1,x2,y1,y2,label_idx):
    def _get_image(img_path):
        img = Image.open(img_path)
        img = img.crop((x1,y1,x2,y2))
        img = img.resize((width, height))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255.
        return img
    def _get_pred_res(img):
        prediction = model.predict(img)
        res_class=np.argmax(prediction)
        return res_class

    w=x2-x1
    h=y2-y1
    if os.path.isdir(os.path.join(img_path ,os.listdir(img_path)[0])):
        for img_dir in os.listdir(img_path):
            for img_name in os.listdir(os.path.join(img_path,img_dir)):
                img = _get_image(os.path.join(img_path,img_dir,img_name))
                res_class = _get_pred_res(img)
                y_true.append(label_idx)
                y_pred.append(res_class)
        
    else:
        for img_name in os.listdir(img_path):
            img = _get_image(os.path.join(img_path,img_name))
            res_class = _get_pred_res(img)
            y_true.append(label_idx)
            y_pred.append(res_class)
    
    return y_true, y_pred

#labels = list(label_map.keys())
#print(labels)

labels = dict()
idx=0
for key in label_map.keys():
    labels[key]=idx
    idx+=1
print(labels)

for k,v in t_config.items():
    x1 = v['x1']
    x2 = v['x2']
    y1 = v['y1']
    y2 = v['y2']
    pos_case = v['pos_case']
    ng_case = v['ng_case']
    position = k
    #print(position) 
    #print(labels[position])

    if len(os.listdir(pos_case))>0:
        y_true,y_pred = pred(pos_case,y_true,y_pred,x1,x2,y1,y2,labels[position+'_ok'])
    else:
        y_true.append(labels[position+'_ok'])
        y_pred.append(labels[position+'_ok'])
    print(ng_case)
    if len(os.listdir(ng_case[0])) > 0:
        for ng_path in ng_case:
            y_true,y_pred = pred(ng_path,y_true,y_pred,x1,x2,y1,y2,labels[position+'_ng'])
    else:
        y_true.append(labels[position+'_ng'])
        y_pred.append(labels[position+'_ng'])

cnf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
print(cnf_matrix)

accuracy = np.trace(cnf_matrix) / float(np.sum(cnf_matrix))
misclass = 1 - accuracy
print('accuracy',accuracy)
print('misclass',misclass)


plot_confusion_matrix(cnf_matrix, title, target_names=labels)
#from sklearn.metrics import classification_report
#print(classification_report(y_true, y_pred, target_names=labels))
