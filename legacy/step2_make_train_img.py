import sys


def make_ng_img(img_generator, roi_coordinate, ng_case, img_ct_train, img_save_path):
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
        img_ct_ng_train=int(ratio_img_ct*img_ct_train)
        ng_list = os.listdir(ng_path)
        img_ct_per_ng_case=int(img_ct_ng_train/len(ng_list))
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
                        tmp.save(os.path.join(img_save_path,'{0:06d}.png'.format(ng_img_ct)))
                    elif img_ct_per_ng_case <= img_ct:
                        break
                    else:
                        crop_img = x[0][y1: y1 + h, x1: x1 + w]
                        im = Image.fromarray(crop_img.astype('uint8'),'RGB')
                        im.save(os.path.join(img_save_path,'{0:06d}.png'.format(ng_img_ct)))
                    img_ct+=1
                    ng_img_ct+=1
        else:
            for img_name in random.sample(os.listdir(ng_path),img_ct_ng_train):
                tmp = Image.open(os.path.join(ng_path,img_name))
                tmp = tmp.crop((x1,y1,x2,y2))
                tmp.save(os.path.join(img_save_path,'{0:06d}.png'.format(ng_img_ct)))
                ng_img_ct += 1


def run(config_path):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp.read())

    position = list(config.keys())[0]
    pos_sample_ratio = config[position]['pos_sample_ratio']
    valid_split_ratio = config[position]['valid_split_ratio']
    prod = config[position]['prod']
    x1 = config[position]['x1']
    x2 = config[position]['x2']
    y1 = config[position]['y1']
    y2 = config[position]['y2']
    h = config[position]['height']
    w = config[position]['width']
    ng_list = config[position]['ng_case_list']
    pos_path = config[position]['pos_case']

    total_pos_sample = len(os.listdir(pos_path))
    img_ct_ok_train = int(pos_sample_ratio*total_pos_sample)
    img_ct_ok_valid = int(img_ct_ok_train*valid_split_ratio)

    img_ct_break=img_ct_ok_train+img_ct_ok_valid
    img_ct=0

    os.makedirs(os.path.join(prod,'train'), exist_ok=True)
    os.makedirs(os.path.join(prod,'train',position+'_ok'), exist_ok=True)
    os.makedirs(os.path.join(prod,'valid'), exist_ok=True)
    os.makedirs(os.path.join(prod,'valid',position+'_ok'), exist_ok=True)
    for img_name in os.listdir(pos_path):
        if img_ct_break <= img_ct:
            break
        tmp = Image.open(os.path.join(pos_path,img_name))
        tmp = tmp.crop((x1,y1,x2,y2))
        if img_ct_ok_train > img_ct:
            tmp.save(os.path.join(prod,'train',position+'_ok/{0:06d}.png'.format(img_ct)))
        else:
            tmp.save(os.path.join(prod,'valid',position+'_ok/{0:06d}.png'.format(img_ct)))
        img_ct+=1
    
    img_generator = ImageDataGenerator(rotation_range = 2, #5도
                                             width_shift_range = 0.02,#0.02
                                             height_shift_range = 0.02,#0.02
                                             shear_range = 0., #0.2
                                             zoom_range = 0., #0.2
                                             horizontal_flip = False,
                                             vertical_flip = False,
                                             brightness_range = None,
                                             validation_split = 0.)


    ng_case=dict()
    roi_coordinate=dict()
    roi_coordinate['x1']=x1
    roi_coordinate['x2']=x2
    roi_coordinate['y1']=y1
    roi_coordinate['y2']=y2
    roi_coordinate['height']=h
    roi_coordinate['width']=w
    if len(os.listdir(ng_list[0])) > 0:
        for ng_path in ng_list:
            ng_case[ng_path]=len(os.listdir(ng_path))
        os.makedirs(os.path.join(prod,'train'), exist_ok=True)
        os.makedirs(os.path.join(prod,'train',position+'_ng'), exist_ok=True)
        os.makedirs(os.path.join(prod,'valid'), exist_ok=True)
        os.makedirs(os.path.join(prod,'valid',position+'_ng'), exist_ok=True)

        train_img_save_path=os.path.join(prod,'train',position+'_ng')
        valid_img_save_path=os.path.join(prod,'valid',position+'_ng')
        make_ng_img(img_generator, roi_coordinate, ng_case, img_ct_ok_train, train_img_save_path)
        make_ng_img(img_generator, roi_coordinate, ng_case, img_ct_ok_valid, valid_img_save_path)
    else:
        print("*"*50)
        print(config_path)
        print("NG 이미지가 없습니다.\n수동 으로 다른 제품 이미지를 넣어야 합니다.")
        print("넣어야 할 이미지 장수\tNG Train: ",img_ct_ok_train,"NG Valid:",img_ct_ok_valid)
        print("*"*50)



if len(sys.argv) == 1:
    print('설정 파일 경로를 입력 하세요.')
    sys.exit()
else:
    import os
    import yaml
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    from keras.preprocessing.image import ImageDataGenerator, img_to_array

config_path = sys.argv[1]
if os.path.isfile(config_path):
    run(config_path)
else:
    if os.path.isdir(config_path):
        for config in os.listdir(config_path):
            run(os.path.join(config_path,config))
    else:
        print('설정 파일 경로를 입력 하세요.')
        sys.exit()


