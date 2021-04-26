import sys
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, Activation, MaxPooling2D, Input, GlobalAveragePooling2D
from keras import applications
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras import applications, optimizers, Model
import numpy as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import yaml
from datetime import datetime
import os
import matplotlib.pyplot as plt
import math
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

def _print_layer_trainable(base_model):
    for layer in base_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

def train_run(project, prod,config_aug):
    if os.path.isdir(os.path.join(project,prod,'train')) is False:
        print("입력된 제품은 존재하지 않습니다.")
        return False

    WIDTH = config_aug['WIDTH']
    HEIGHT = config_aug['HEIGHT']
    NUM_EPOCHS = config_aug['NUM_EPOCHS']
    EARLY_STOP_PATIENCE = config_aug['EARLY_STOP_PATIENCE']
    lr = config_aug['LEARNING_RATE']
    decay = config_aug['LEARNING_DACAY_RATE']
    OPTIMIZER = 'rmsprop'
    rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=decay)
    NUM_CLASSES = len(os.listdir(os.path.join(project,prod,'train')))

    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(project,prod,'model'), exist_ok=True)
    MODEL_NAME = 'train_'+prod+'_'+OPTIMIZER+'_'+cur_time+'.h5'
    MODEL_SAVE_PATH = os.path.join(project,prod,'model',MODEL_NAME)
    model_config_path = os.path.join(project,prod,'model_config')
    os.makedirs(model_config_path, exist_ok=True)

    config = dict()
    config['prod'] = prod
    config['width'] = WIDTH
    config['height'] = HEIGHT
    config['optimizer'] = OPTIMIZER
    config['model_save_path'] = MODEL_SAVE_PATH
    config['model_name'] = MODEL_NAME
    config['NUM_EPOCHS'] = NUM_EPOCHS
    config['EARLY_STOP_PATIENCE'] = EARLY_STOP_PATIENCE
    config['LEARNING_RATE'] = lr
    config['LEARNING_DACAY_RATE'] = decay
    
    train_len = []
    for train_datas in os.listdir(os.path.join(project,prod,'train')):
        train_len.append(len(os.listdir(os.path.join(project,prod,'train',train_datas))))
    batch_size_train = min(train_len)

    valid_len = []
    for valid_datas in os.listdir(os.path.join(project,prod,'valid')):
        valid_len.append(len(os.listdir(os.path.join(project,prod,'valid',valid_datas))))
    batch_size_valid = min(valid_len)
    
    if batch_size_train >= 32:
        batch_size_train = 32
    elif batch_size_train >= 16:
        batch_size_train = 16
    elif batch_size_train >= 8:
        batch_size_train = 8
    else:
        batch_size_train = 1
    
    if batch_size_valid >= 32:
        batch_size_valid = 32
    elif batch_size_valid >= 16:
        batch_size_valid = 16
    elif batch_size_valid >= 8:
        batch_size_valid = 8
    else:
        batch_size_valid = 1

    target_size = (WIDTH,HEIGHT)
    train_datagen = ImageDataGenerator(rescale = 1./255)
    valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(project,prod,'train'),
        target_size=target_size,
        batch_size=batch_size_train,
        class_mode='categorical')

    valid_gen = valid_datagen.flow_from_directory(
        os.path.join(project,prod,'valid'),
        target_size=target_size,
        batch_size=batch_size_valid,
        class_mode='categorical')
    
    config['label_map'] = (train_gen.class_indices)
    with open(os.path.join(model_config_path, 'train_'+prod+'_'+OPTIMIZER+'_'+cur_time+'.yaml'), 'w') as fp:
        yaml.dump(config, fp)

    input_tensor = Input(shape=(WIDTH, HEIGHT, 3))
    base_model = applications.VGG16(weights='imagenet',include_top=False,input_tensor=input_tensor)
    base_model.trainable=False
    for layer in base_model.layers:
        layer.trainable=False
    add_model = Sequential()
    add_model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=base_model.output_shape[1:]))
    add_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    add_model.add(Flatten())
    add_model.add(Dense(32, activation='relu'))
    add_model.add(Dropout(0.5))
    add_model.add(Dense(NUM_CLASSES, activation='softmax'))
    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    _print_layer_trainable(model)

    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    ear = EarlyStopping(monitor='val_loss',
                        patience=EARLY_STOP_PATIENCE,
                        mode='auto')

    mcp = ModelCheckpoint(MODEL_SAVE_PATH,
                          monitor='val_loss',
                          save_best_only=True,
                          mode='auto')

    hist = model.fit_generator(
        train_gen,
        epochs=NUM_EPOCHS,
        validation_data=valid_gen,
        callbacks=[ear, mcp],
        steps_per_epoch=train_gen.samples//train_gen.batch_size,
        validation_steps=math.ceil(valid_gen.samples//valid_gen.batch_size)
    )

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='loss')
    acc_ax.plot(hist.history['acc'], 'b', label='acc')
    
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')
    
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='upper right')

    hist_save_path = "train_"+prod+"_"+OPTIMIZER+'_'+cur_time+".png"
    plt.savefig(os.path.join(project,hist_save_path), bbox_inces='tight', pad_inches=0, dpi=100)
    print("\n학습 그래프 저장완료 | 저장 위치->",os.path.join(project,hist_save_path))
    print("\n")
