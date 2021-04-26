from PIL import Image, ImageDraw
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
import numpy as np
from termcolor import colored
import os
import tensorflow as tf
import yaml


#graph = tf.get_default_graph()


class Inference_Keras_Vision:
    rounds = 9
    def __init__(self):
        self.models = dict()
        prod_list = list()
        self.t_config = dict()
        for cfg_name in os.listdir('live'):
            with open(os.path.join('live',cfg_name), 'r') as fp:
                m_config = yaml.load(fp.read())
            prod = m_config['prod']
            self.models[prod] = load_model(m_config['model_save_path'], compile=False)
            self.models[prod]._make_predict_function()
            point_config = dict()
            for config_name in os.listdir(os.path.join(prod,'roi_config')):
                with open(os.path.join(prod,'roi_config',config_name), 'r') as fp:
                    config = yaml.load(fp.read())
                t_sub_config = dict()
                t_sub_config['pos_case'] = config[list(config.keys())[0]]['pos_case']
                t_sub_config['ng_case'] = config[list(config.keys())[0]]['ng_case_list']
                t_sub_config['x1'] = config[list(config.keys())[0]]['x1']
                t_sub_config['x2'] = config[list(config.keys())[0]]['x2']
                t_sub_config['y1'] = config[list(config.keys())[0]]['y1']
                t_sub_config['y2'] = config[list(config.keys())[0]]['y2']
                point_config[list(config.keys())[0]] = t_sub_config
            self.t_config[prod] = point_config
            self.t_config[prod]['label_map'] = m_config['label_map']
            self.t_config[prod]['width'] = m_config['width']
            self.t_config[prod]['height'] = m_config['height']

    def _prep_from_image(self, img_path, prod, point):
        img = np.array(load_img(img_path))
        x1 = self.t_config[prod][point]['x1']
        x2 = self.t_config[prod][point]['x2']
        y1 = self.t_config[prod][point]['y1']
        y2 = self.t_config[prod][point]['y2']
        width = self.t_config[prod]['width']
        height = self.t_config[prod]['height']

        im = Image.fromarray(img)
        im = im.crop((x1,y1,x2,y2))
        im = im.resize((width, height))
        img = np.array(im)
        img = np.expand_dims(img, axis=0)
        return self._gen_augment_arrays(img, np.array([]))


    def _gen_augment_arrays(self, array, label):
        img_generator = ImageDataGenerator(
                                            rotation_range = 2, #5도
                                            width_shift_range = 0.02,#0.02
                                            height_shift_range = 0.02,#0.02
                                            shear_range = 0., #0.2
                                            zoom_range = 0., #0.2
                                            horizontal_flip = False,
                                            vertical_flip = False,
                                            brightness_range = None,
                                            validation_split = 0.
                                        )
        array_augs, label_augs = next(img_generator.flow(np.tile(array,(self.rounds,1,1,1)), np.tile(label,(self.rounds,1)), batch_size=self.rounds))
        #idx=0
        for array_aug, label_aug in zip(array_augs, label_augs):
            yield array_aug, label_aug
            #idx+=1
            #im = Image.fromarray(array_aug.astype('uint8'),'RGB')
            #im.save('{0:06d}.png'.format(idx))


    def run_inference_on_image(self, img_path, prod, point):
        print('Predictiong on image in: ',colored(img_path,'yellow'))
        print('Prod: ',colored(prod,'blue'),'Point: ',colored(point,'blue'))
        aug_gen = self._prep_from_image(os.path.expanduser(img_path), prod, point)
        predicted = self._predict(aug_gen, prod)
        PASS = 0.
        FAIL = 0.
        TOTAL = 0.
        for pred_idx in predicted:
            predicted_info = [k for k,v in self.t_config[prod]['label_map'].items() if v == pred_idx][0]
            predicted_point = predicted_info.split('_')[0]
            predicted_result = predicted_info.split('_')[1]
            if point == predicted_point and predicted_result == 'ok':
                PASS += 1.
            else:
                FAIL += 1.
            TOTAL += 1.
        PASS_PER = PASS/TOTAL * 100.
        FAIL_PER = FAIL/TOTAL * 100.
        ACCURACY = 0.
        if PASS_PER > FAIL_PER:
            PRED_RESULT = 'ok'
            ACCURACY = PASS_PER
        else:
            PRED_RESULT = 'ng'
            ACCURACY = 100.-FAIL_PER
        return PRED_RESULT, ACCURACY


    def _predict(self, aug_gen, prod):
        predicted = []
        for img, _ in aug_gen:
            img = np.expand_dims(img, axis=0)
            img = img/255.
            #global graph
            #with graph.as_default():
            predicted.append(np.argmax(self.models[prod].predict(img)))
        return predicted


if __name__ == "__main__":
    img_path='/home/gtpark/Desktop/1_1_ng.jpg'
    prod = 'HCYB'
    point = '1'
    
    inference = Inference_Keras_Vision()
    pred_result, pass_per, fail_per = inference.run_inference_on_image(img_path, prod, point)
    print('pred_result',pred_result)
    print('pass_per',pass_per)
    print('fail_per',fail_per)

