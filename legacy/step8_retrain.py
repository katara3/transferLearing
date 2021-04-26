import os
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, img_to_array

t_prod = 'C300'

for cfg_name in os.listdir('live'):
    with open(os.path.join('live',cfg_name), 'r') as fp:
        config = yaml.load(fp.read())
    prod = config['prod']
    if prod == t_prod:
    	model = load_model(config['model_save_path'])
    	WIDTH = config['width']
    	HEIGHT = config['height']
    	break

os.makedirs(os.path.join(prod,'model_config'), exist_ok=True)
config_path = os.path.join(prod,'model_config')
BATCH_SIZE = 16
EARLY_STOP_PATIENCE = 2
NUM_EPOCHS = 50
OPTIMIZER = 'rmsprop'
cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = os.path.join(prod,'model','retrain_'+prod+'_'+OPTIMIZER+'_'+cur_time+'.h5')
config['model_save_path'] = MODEL_SAVE_PATH

with open(os.path.join(config_path, 'retrain_'+prod+'_'+OPTIMIZER+'_'+cur_time+'.yaml'), 'w') as fp:
    yaml.dump(config, fp)


target_size = (WIDTH, HEIGHT)
train_datagen = ImageDataGenerator(rescale = 1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(prod,'train'),
    target_size=target_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

valid_gen = valid_datagen.flow_from_directory(
    os.path.join(prod,'valid'),
    target_size=target_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical')
    
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

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_acc'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.savefig("retrain_"+prod+"_"+OPTIMIZER+'_'+cur_time+".png",bbox_inces='tight', pad_inches=0, dpi=100)

