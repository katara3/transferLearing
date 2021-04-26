import os
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, img_to_array
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



def retrain_run(project,prod,model_save_path,config_aug):
    if os.path.isdir(os.path.join(project,prod,'train')) is False:
        print("입력된 제품은 존재하지 않습니다.")
        return False
    try:
        model = load_model(model_save_path)
    except:
        print("입력된 모델은 존재하지 않습니다.")
        return False

    BATCH_SIZE = 16
    OPTIMIZER = 'rmsprop'
    WIDTH = config_aug['WIDTH']
    HEIGHT = config_aug['HEIGHT']
    NUM_EPOCHS = config_aug['NUM_EPOCHS']
    EARLY_STOP_PATIENCE = config_aug['EARLY_STOP_PATIENCE']
    ######################################
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(project,prod,'model'), exist_ok=True)
    MODEL_NAME = 'retrain_'+prod+'_'+OPTIMIZER+'_'+cur_time+'.h5'
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

    target_size = (WIDTH, HEIGHT)
    train_datagen = ImageDataGenerator(rescale = 1./255)
    valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(project,prod,'train'),
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    valid_gen = valid_datagen.flow_from_directory(
        os.path.join(project,prod,'valid'),
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    config['label_map'] = (train_gen.class_indices)
    with open(os.path.join(model_config_path, 'retrain_'+prod+'_'+OPTIMIZER+'_'+cur_time+'.yaml'), 'w') as fp:
        yaml.dump(config, fp)

    ear = EarlyStopping(monitor='acc',
                        patience=EARLY_STOP_PATIENCE,
                        mode='max')

    mcp = ModelCheckpoint(MODEL_SAVE_PATH,
                          monitor='acc',
                          save_best_only=True,
                          mode='max')

    hist = model.fit_generator(
        train_gen,
        epochs=NUM_EPOCHS,
        validation_data=valid_gen,
        callbacks=[ear, mcp],
        steps_per_epoch=train_gen.samples//BATCH_SIZE,
        validation_steps=valid_gen.samples//BATCH_SIZE
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

    hist_save_path = "retrain_"+prod+"_"+OPTIMIZER+'_'+cur_time+".png"
    plt.savefig(os.path.join(project,hist_save_path), bbox_inces='tight', pad_inches=0, dpi=100)
    print("\n학습 그래프 저장완료 | 저장 위치->",os.path.join(project,hist_save_path))
    print("\n")

