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
from gpuinfo import GPUInfo

gpu_info = (GPUInfo.get_info())
num_list = [0,1,2,3]
gpu_dict = dict()
for x in num_list:
    gpu_dict[x] = gpu_info[2][x]

def find_gpu(x):
    return gpu_dict[x]
key_min = min(gpu_dict.keys(), key=(lambda k: gpu_dict[k]))
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(key_min)


def _plot_confusion_matrix(project, cm, title, target_names=None, cmap=None, normalize=True, labels=True):
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
    cm_save_path = "valid_"+title+'_'+cur_time+".png"
    plt.savefig(os.path.join(project,cm_save_path),bbox_inces='tight', pad_inches=0, dpi=100)
    print("\nConfusion matrix 결과 저장 완료 | 위치 ->",os.path.join(project,cm_save_path))
    print("\n")


def _pred(model,img_path,width,height,label_idx):
    def _get_image(img_path):
        img = Image.open(img_path)
        img = img.resize((width, height))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255.
        return img

    def _get_pred_res(img):
        prediction = model.predict(img)
        res_class=np.argmax(prediction)
        return res_class
    
    y_true = []
    y_pred = [] 
    for img_name in os.listdir(img_path):
        img = _get_image(os.path.join(img_path,img_name))
        res_class = _get_pred_res(img)
        y_true.append(label_idx)
        y_pred.append(res_class)
    return y_true, y_pred


def valid_run(project, model_config_path):
    with open(model_config_path, 'r') as fp:
        config = yaml.safe_load(fp.read())
    width = config['width']
    height = config['height']
    model=load_model(config['model_save_path'])
    label_map = config['label_map']
    prod = config['prod']
    title = 'Product: '+prod.replace('/','')


    y_true=[]
    y_pred=[]
    
    labels = dict()
    idx=0
    train_list = os.listdir( os.path.join(project, prod, 'train'))
    for key in label_map.keys():
        if key in train_list:
            labels[key]=idx
        idx+=1

    for point_dir in train_list:
        img_path = os.path.join(project, prod, 'train', point_dir)
        tmp_y_true, tmp_y_pred = [], []
        tmp_y_true, tmp_y_pred = _pred(model,img_path,width,height,labels[point_dir])
        y_true.extend(tmp_y_true)
        y_pred.extend(tmp_y_pred)
    
    for point_dir in os.listdir( os.path.join(project, prod, 'valid')):
        img_path = os.path.join(project, prod, 'valid', point_dir)
        tmp_y_true, tmp_y_pred = [], []
        tmp_y_true, tmp_y_pred = _pred(model,img_path,width,height,labels[point_dir])
        y_true.extend(tmp_y_true)
        y_pred.extend(tmp_y_pred)

    cnf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

    accuracy = np.trace(cnf_matrix) / float(np.sum(cnf_matrix))
    misclass = 1 - accuracy
    print("\n검증 완료")
    print('accuracy',accuracy)
    print('misclass',misclass)
    _plot_confusion_matrix(project, cnf_matrix, title, target_names=labels)
    from sklearn.metrics import classification_report
    print("F-1 Score 리포트")
    print(classification_report(y_true, y_pred, target_names=labels))
    print("\n")
