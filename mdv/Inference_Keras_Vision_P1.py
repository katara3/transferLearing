from PIL import Image, ImageDraw
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
import numpy as np
from termcolor import colored
import os
import tensorflow as tf
import yaml



class Inference_Keras_Vision:
    rounds = 9

    def __init__(self):
        self.models = dict()
        prod_list = list()
        self.t_config = dict()
        
        for cfg_name in os.listdir(os.path.join('live','config')):
            with open(os.path.join('live','config',cfg_name), 'r') as fp:
                m_config = yaml.safe_load(fp.read())
            prod = m_config['prod']
            self.models[prod] = load_model(m_config['model_save_path'], compile=False)
            self.models[prod]._make_predict_function()
            
            point_config = dict()
            for point in m_config['roi']:
                t_sub_config = dict()
                t_sub_config['x1'] = m_config['roi'][point]['x1']
                t_sub_config['x2'] = m_config['roi'][point]['x2']
                t_sub_config['y1'] = m_config['roi'][point]['y1']
                t_sub_config['y2'] = m_config['roi'][point]['y2']
                point_config[point] = t_sub_config
            self.t_config[prod] = point_config
            self.t_config[prod]['inference_threshold'] = m_config['inference_threshold']
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
        ###########################################################################
        home = os.path.expanduser('~')
        config_path = os.path.join(home, '.mdv', 'config.yaml')
        if os.path.isfile(config_path):
            with open(config_path, 'r') as fp:
                config_aug = yaml.load(fp.read())
        img_generator = ImageDataGenerator(
                                            rotation_range = config_aug['rotation_range'], #5도
                                            width_shift_range = config_aug['width_shift_range'],#0.02
                                            height_shift_range = config_aug['height_shift_range'],#0.02
                                            shear_range = config_aug['shear_range'], #0.2
                                            zoom_range = config_aug['zoom_range'], #0.2
                                            horizontal_flip = config_aug['horizontal_flip'],
                                            vertical_flip = config_aug['vertical_flip'],
                                            brightness_range = config_aug['brightness_range'],
                                            validation_split = config_aug['validation_split']
                                        )

        array_augs, label_augs = next(img_generator.flow(np.tile(array,(self.rounds,1,1,1)), np.tile(label,(self.rounds,1)), batch_size=self.rounds))
        for array_aug, label_aug in zip(array_augs, label_augs):
            yield array_aug, label_aug


    def run_inference_on_image(self, img_path, prod, point):
        print('Predictiong on image in: ',colored(img_path,'yellow'))
        print('Prod: ',colored(prod,'blue'),'Point: ',colored(point,'blue'))
        aug_gen = self._prep_from_image(os.path.expanduser(img_path), prod, point)
        predicted = self._predict(aug_gen, prod)
        OK = 0.
        NG = 0.
        FAIL = 0.
        TOTAL = 0.
        for pred_idx in predicted:
            predicted_info = [k for k,v in self.t_config[prod]['label_map'].items() if v == pred_idx][0]
            predicted_point = predicted_info.split('_')[0]
            predicted_result = predicted_info.split('_')[1]
            print(predicted_info)
            point_str = str(point)
            if point_str == predicted_point and predicted_result == 'ok':
                OK += 1.
            elif point_str == predicted_point and predicted_result == 'ng':
                NG += 1. 
            else:
                print('fail',point_str,predicted_point,predicted_result)
                FAIL += 1.
		
            TOTAL += 1.

        OK_PER = OK/TOTAL * 100.
        NG_PER = NG/TOTAL * 100.
        FAIL_PER = FAIL/TOTAL * 100.
        if FAIL_PER > OK_PER+NG_PER:
            PRED_RESULT = 'fail'
            ACCURACY = 0.
        else:
            if int((self.rounds-FAIL)*(float(self.t_config[prod]['inference_threshold'])/100.)) <= OK or int((self.rounds-FAIL)*(1-float(self.t_config[prod]['inference_threshold'])/100.)) >= OK:
                if OK_PER >= NG_PER:
                    PRED_RESULT = 'ok'
                    ACCURACY = round(OK_PER/(OK_PER+NG_PER)*100.,2)
                else:
                    PRED_RESULT = 'ng'
                    ACCURACY = round(NG_PER/(OK_PER+NG_PER)*100.,2)
            else:
                PRED_RESULT = 'fail'
                ACCURACY = 0.

        return PRED_RESULT, ACCURACY


    def _predict(self, aug_gen, prod):
        predicted = []
        for img, _ in aug_gen:
            img = np.expand_dims(img, axis=0)
            img = img/255.
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