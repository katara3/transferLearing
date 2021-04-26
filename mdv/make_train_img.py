import sys
import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from keras.preprocessing.image import ImageDataGenerator, img_to_array


def _make_ng_img(prod, img_generator, roi_coordinate, ng_case, img_ct_train, img_save_path):
    x1=roi_coordinate['x1']
    x2=roi_coordinate['x2']
    y1=roi_coordinate['y1']
    y2=roi_coordinate['y2']
    h=roi_coordinate['height']
    w=roi_coordinate['width']

    total_img_ct = sum(ng_case.values())
    ng_img_ct=0

    for ng_path, img_ct in ng_case.items():
        ratio_img_ct = img_ct/(total_img_ct)
        img_ct_ng_train = int(ratio_img_ct*img_ct_train)
        ng_list = os.listdir(ng_path)
        img_ct_per_ng_case = int(img_ct_ng_train/len(ng_list))
        if img_ct_per_ng_case > 0:
            for img_name in ng_list:
                tmp = Image.open(os.path.join(ng_path,img_name))
                tmp = np.array(tmp)
                tmp = np.expand_dims(tmp, axis=0)

                img_ct=0
                for x in img_generator.flow(tmp, batch_size=1):
                    if img_ct == 0:
                        tmp = Image.fromarray(tmp[0].astype('uint8'),'RGB')
                        tmp = tmp.crop((x1,y1,x2,y2))
                        tmp.save(os.path.join(img_save_path, prod+'_{0:06d}.png'.format(ng_img_ct)))
                    elif img_ct_per_ng_case <= img_ct:
                        break
                    else:
                        crop_img = x[0][y1: y1 + h, x1: x1 + w]
                        im = Image.fromarray(crop_img.astype('uint8'),'RGB')
                        im.save(os.path.join(img_save_path, prod+'_{0:06d}.png'.format(ng_img_ct)))
                    img_ct += 1
                    ng_img_ct += 1
        else:
            for img_name in random.sample(os.listdir(ng_path),img_ct_ng_train):
                tmp = Image.open(os.path.join(ng_path,img_name))
                tmp = tmp.crop((x1,y1,x2,y2))
                tmp.save(os.path.join(img_save_path, prod+'_{0:06d}.png'.format(ng_img_ct)))
                ng_img_ct += 1


def _make(project, config_path,config_aug):
    print('\n학습 데이터 생성을 시작합니다. roi_config_path: '+config_path)
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp.read())
    
    point = list(config.keys())[0]
    pos_sample_ratio = config[point]['pos_sample_ratio']
    valid_split_ratio = config[point]['valid_split_ratio']
    prod = config[point]['prod']
    x1 = config[point]['x1']
    x2 = config[point]['x2']
    y1 = config[point]['y1']
    y2 = config[point]['y2']
    h = config[point]['height']
    w = config[point]['width']
    ng_list = config[point]['ng_case_list']
    ng_case_ct = config[point]['ng_case_ct']
    pos_path = config[point]['pos_case']
    
    total_pos_sample = len(os.listdir(pos_path))
    img_ct_ok_train = int(pos_sample_ratio * total_pos_sample)
    img_ct_ok_valid = int(img_ct_ok_train * valid_split_ratio)

    img_ct_break = img_ct_ok_train+img_ct_ok_valid
    img_ct=0

    os.makedirs(os.path.join(project, prod), exist_ok=True)
    os.makedirs(os.path.join(project, prod,'train'), exist_ok=True)
    os.makedirs(os.path.join(project, prod,'train',point+'_ok'), exist_ok=True)
    os.makedirs(os.path.join(project, prod,'valid'), exist_ok=True)
    os.makedirs(os.path.join(project, prod,'valid',point+'_ok'), exist_ok=True)
    os.makedirs(os.path.join(project, prod,'train_sample'), exist_ok=True)

    is_sample_save = False
    for img_name in os.listdir(pos_path):
        if img_ct_break <= img_ct:
            break
        tmp = Image.open(os.path.join(pos_path,img_name))
        tmp = tmp.crop((x1,y1,x2,y2))
        if is_sample_save == False:
            tmp.save(os.path.join(project, prod,'train_sample',point+'.png'))
            is_sample_save = True
        if img_ct_ok_train > img_ct:
            tmp.save(os.path.join(project,prod,'train',point+'_ok', prod+'_{0:06d}.png'.format(img_ct)))
        else:
            tmp.save(os.path.join(project,prod,'valid',point+'_ok', prod+'_{0:06d}.png'.format(img_ct)))
        img_ct+=1

    img_generator = ImageDataGenerator(
        rotation_range=config_aug['rotation_range'],  # 5도
        width_shift_range=config_aug['width_shift_range'],  # 0.02
        height_shift_range=config_aug['height_shift_range'],  # 0.02
        shear_range=config_aug['shear_range'],  # 0.2
        zoom_range=config_aug['zoom_range'],  # 0.2
        horizontal_flip=config_aug['horizontal_flip'],
        vertical_flip=config_aug['vertical_flip'],
        brightness_range=config_aug['brightness_range'],
        validation_split=config_aug['validation_split']
    )

    ng_case=dict()
    roi_coordinate=dict()
    roi_coordinate['x1']=x1
    roi_coordinate['x2']=x2
    roi_coordinate['y1']=y1
    roi_coordinate['y2']=y2
    roi_coordinate['height']=h
    roi_coordinate['width']=w
    if ng_case_ct > 0:
        for ng_path in ng_list:
            ng_case[ng_path] = len(os.listdir(ng_path))
        os.makedirs(os.path.join(project,prod,'train'), exist_ok=True)
        os.makedirs(os.path.join(project,prod,'valid'), exist_ok=True)
        os.makedirs(os.path.join(project,prod,'train',point+'_ng'), exist_ok=True)
        os.makedirs(os.path.join(project,prod,'valid',point+'_ng'), exist_ok=True)
		
        train_img_save_path=os.path.join(project,prod,'train',point+'_ng')
        valid_img_save_path=os.path.join(project,prod,'valid',point+'_ng')
        _make_ng_img(prod,img_generator, roi_coordinate, ng_case, img_ct_ok_train, train_img_save_path)
        _make_ng_img(prod,img_generator, roi_coordinate, ng_case, img_ct_ok_valid, valid_img_save_path)

def make_run(project, config_path,config_aug):
    if os.path.isfile(config_path):
        _make(project, config_path,config_aug)
    else:
        if os.path.isdir(config_path):
            for config in os.listdir(config_path):
                _make(project, os.path.join(config_path, config),config_aug)
        else:
            print("제품명 또는 ROI설정 파일 경로가 잘못되었습니다.")
