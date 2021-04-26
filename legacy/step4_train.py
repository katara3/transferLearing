import sys


if len(sys.argv) == 1:
    print('학습할 제품명을 입력하세요')
    sys.exit()
else:
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

    prod = sys.argv[1]
    prod = prod.replace('/','')

def print_layer_trainable(base_model):
    for layer in base_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))


'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.55
set_session(tf.Session(config=config))
'''

############수정할 부분####################
WIDTH = 220
HEIGHT = 220
BATCH_SIZE = 16
NUM_CLASSES = len(os.listdir(os.path.join(prod,'roi_config')))*2
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 2
OPTIMIZER = 'rmsprop'

###########################################

cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join(prod,'model'), exist_ok=True)
MODEL_SAVE_PATH = os.path.join(prod,'model',prod+'_'+OPTIMIZER+'_'+cur_time+'.h5')

config_path = os.path.join(prod,'model_config')
os.makedirs(os.path.join(prod,'model_config'), exist_ok=True)
config = dict()
config['width'] = WIDTH
config['height'] = HEIGHT
config['model_save_path'] = MODEL_SAVE_PATH
config['prod'] = prod
config['optimizer'] = OPTIMIZER

target_size = (WIDTH,HEIGHT)
train_datagen = ImageDataGenerator(rescale = 1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

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

train_label_map = (train_gen.class_indices)
valid_label_map = (valid_gen.class_indices)
print("*"*50)
print(train_label_map)
config['label_map'] = train_label_map 
with open(os.path.join(config_path, 'train_'+prod+'_'+OPTIMIZER+'_'+cur_time+'.yaml'), 'w') as fp:
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
#add_model.add(Dense(16, activation='relu'))
add_model.add(Dropout(0.5))
add_model.add(Dense(NUM_CLASSES, activation='softmax'))
model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
print_layer_trainable(model)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

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

plt.savefig("train_"+prod+"_"+OPTIMIZER+'_'+cur_time+".png",bbox_inces='tight', pad_inches=0, dpi=100)

