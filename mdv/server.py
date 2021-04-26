# coding:utf-8
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, make_response, jsonify
import time
import socket
from werkzeug.utils import secure_filename
import os

from keras.models import load_model
import yaml
from PIL import Image
import numpy as np
import tensorflow as tf
from mdv.Inference_Keras_Vision_P1 import Inference_Keras_Vision
from termcolor import colored
import json
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


def start_server():
    app = Flask(__name__)
    app.config['UPLOAD_VISION'] = '.'
    inferenceKeras_vision = Inference_Keras_Vision()

    @app.route('/TestVision/v1', methods=['POST'])
    def TestVision1():
        #print('***** Vision Flask function start TestVision1 *****')
        try:
            arg_json = request.form.to_dict()  # parameters json
        except:
            return make_response(jsonify(msg="Vision not allowed dictionary"), 401)
        
        start_time = time.time()
        # if exists upload_file
        if 'UPLOAD_FILE' in request.files:
            # Get client file Name
            file = request.files['UPLOAD_FILE']
            #print('upload_file OK')
            filename = secure_filename(file.filename)
            #print('filename : ' + filename)
            filename_woext = filename.rsplit('.', 1)[0]  # 확장자제외 filename
            #print('received filename = ' + filename)
            
            if "PROD" in arg_json:
                prod = arg_json['PROD']
            else:
                return make_response(jsonify(msg="Parameter prod is not found"), 401)
            
            if "POINT" in arg_json:
                point = arg_json['POINT']
            else:
                return make_response(jsonify(msg="Parameter point is not found"), 401)

            # Create file name (image)
            f_ext = filename.rsplit('.', 1)[1].lower()
            file_name_data = str(start_time)+'.'+f_ext
            
            try:
                file.save(os.path.join(app.config['UPLOAD_VISION'], file_name_data))
                img_path = app.config['UPLOAD_VISION'] + '/' + file_name_data
                #print('Saving Vision image to local completed  = ' + img_path)
            
            except Exception as ex:
                #print('Error : Saving Vision file and imgPath to local')
                return make_response(jsonify(msg="Saving Vision file and imgPath to local error"), 401)

            try:
                pred_result, accuracy = inferenceKeras_vision.run_inference_on_image(img_path, prod, point)
            except Exception as e:
                print(e)
                return make_response(jsonify(msg="This product or point dose not exist"), 401)
            end_time = time.time()
            result = {
                    'prod': prod,
                    'point': point,
                    'accuracy': accuracy,
                    'pred_result': pred_result,
                    'response_time': end_time-start_time
            }
            try:
                os.remove(img_path)
                #print('File remove completed')
            except:
                return make_response(jsonify(msg="Failed to delete file"), 401)
            
            response = app.response_class(
                response= json.dumps(result),
                status=200,
                mimetype='application/json')
            return response
        else:
            return make_response(jsonify(msg="No upload file"), 401)

    app.run(host='0.0.0.0', debug=False, port=8302, threaded=True)
    
