import os, sys
import yaml
import shutil



def export_run(model_config_path, roi_config_path, inference_threshold):
    with open(model_config_path, 'r') as fp:
        config = yaml.safe_load(fp.read())

    os.makedirs('live', exist_ok=True)
    os.makedirs(os.path.join('live','model'), exist_ok=True)
    os.makedirs(os.path.join('live','config'), exist_ok=True)
    shutil.copy(config['model_save_path'],os.path.join('live','model',config['model_name']))
    root_config=dict()
    root_config['width'] = config['width']
    root_config['height'] = config['height']
    root_config['model_save_path'] = os.path.join('live','model',config['model_name'])
    root_config['label_map'] = config['label_map']
    root_config['prod'] = config['prod']
    root_config['inference_threshold'] = inference_threshold
    prod = root_config['prod']

    
    t_config=dict()
    for config_name in os.listdir(roi_config_path):
        with open(os.path.join(roi_config_path,config_name), 'r') as fp:
            config = yaml.safe_load(fp.read())
        t_sub_config=dict()
        t_sub_config['pos_case'] =  config[list(config.keys())[0]]['pos_case']
        t_sub_config['ng_case'] = config[list(config.keys())[0]]['ng_case_list']
        t_sub_config['ng_case_ct'] = config[list(config.keys())[0]]['ng_case_ct']
        t_sub_config['x1'] = config[list(config.keys())[0]]['x1']
        t_sub_config['x2'] = config[list(config.keys())[0]]['x2']
        t_sub_config['y1'] = config[list(config.keys())[0]]['y1']
        t_sub_config['y2'] = config[list(config.keys())[0]]['y2']
        t_config[list(config.keys())[0]] = t_sub_config
    
    root_config['roi'] = t_config
    with open(os.path.join('live','config',prod+'.yaml'), 'w') as fp:
        yaml.dump(root_config, fp)
