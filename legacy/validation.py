import sys
import os
import requests
from subprocess import Popen
import json

path = ('/home/dde/Desktop/test/image_files')
folder_list = os.listdir(path)
file_path = []
for x in range(0,1):
    PROD = 'C300'
    x = PROD
    point_list = os.listdir(path+'/'+PROD)
    point_list.remove('model')
    point_list.remove('roi_config')

    for y in point_list:
        if y[-1] == 'k':
            continue
        POINT = y
        file_list = os.listdir(path+'/'+PROD+'/'+y+'/OK')
        fail_list_m = []
        fail_list = []
        suc_m = 0
        suc = 0
        error_list = []
        error_list_m = []
        total_m = 0
        total = 0
        fail_list_o = []
        fail_list_n = []
        suc_o = 0
        error_list_o = []
        suc_n = 0

        if POINT == '4':
            g = open('/home/dde/Desktop/test/save_result/{}'.format(PROD+'_4MSCL'),'w')
            f = open('/home/dde/Desktop/test/save_result/{}'.format(PROD+'_4ESCL'),'w')
            g_1 = open('/home/dde/Desktop/test/save_result/{}'.format(PROD+'_4MSCL_only'),'w')
            f_1 = open('/home/dde/Desktop/test/save_result/{}'.format(PROD+'_4ESCL_only'),'w')

        else:
            f = open('/home/dde/Desktop/test/save_result/{}'.format(PROD+'_'+y),'w')
            f_1 = open('/home/dde/Desktop/test/save_result/{}'.format(PROD+'_'+y+'_only'),'w')


        for n,z in enumerate (file_list):
            if y == '4':
                if 'MSCL' in z:
                    print('MSCL',z)
                    POINT = '4MSCL'
                    total_m += 1
                elif 'ESCL' in z:
                    print('ESCL',z)
                    POINT = '4ESCL'
                    total += 1
            else:
                total += 1
            command='''curl -i -X POST -H "Content-Type:multipart/form-data" -F "UPLOAD_FILE=@{}" -F "POINT={}" -F "PROD={}" http://127.0.0.1:8302/TestVision/v1'''.format(path+'/'+PROD+'/'+POINT+'/OK/'+z,POINT,PROD)
            result = os.popen(command).read().strip().split('\n')
            print(result)
            print('data',x,y,z)

            try:
                data = json.loads(result[-1])
            except:
                if POINT == '4MSCL':
                    error_list_m.append(x+'/'+y+'/'+z)
                else:
                    error_list.append(x+'/'+y+'/'+z)
                continue
            if POINT == '4MSCL':
                if data['pred_result'] != 'ng':
                    suc_n += 1
                    if ' NG ' in z:
                        fail_list_m.append('PROD = '+PROD+' , POINT = '+POINT+' , FILE_NAME = '+z)
                    elif ' OK ' in z:
                        suc_m += 1
                else:
                    fail_list_n.append('PROD = ' + PROD + ' , POINT = ' + POINT + ' , FILE_NAME = ' + z)
                    if ' NG ' in z:
                        suc_m += 1

                    elif ' OK ' in z:
                        fail_list_m.append('PROD = '+PROD+' , POINT = '+POINT+' , FILE_NAME = '+z)

            else:
                if data['pred_result'] != 'ng':
                    suc_o += 1
                    if ' NG ' in z:
                        fail_list.append('PROD = '+PROD+' , POINT = '+POINT+' , FILE_NAME = '+z)
                        
                    elif ' OK ' in z:
                        suc += 1

                else:
                    fail_list_o.append('PROD = ' + PROD + ' , POINT = ' + POINT + ' , FILE_NAME = ' + z)
                    if ' NG ' in z:
                        suc += 1

                    elif ' OK ' in z:
                        fail_list.append('PROD = '+PROD+' , POINT = '+POINT+' , FILE_NAME = '+z)


        score = float(suc)/float(total)*100
        score_o = float(suc_o)/float(total)*100

        if POINT[-1] == 'L':
            score_m = float(suc_m)/float(total_m)*100
            score_n = float(suc_n)/float(total_m)*100
            POINT = '4ESCL'
            g.write('PROD = '+PROD+' , POINT = 4MSCL , 성공률 = '+str(score_m)+' , 전체 데이터 = '+str(total_m)+' , 성공 데이터 = '+str(suc_m)+' , 실패 데이터 = '+str(total_m-suc_m)+'\n'+'-'*30+'실패 이미지 리스트'+'-'*30+'\n')
            g_1.write('PROD = '+PROD+' , POINT = 4MSCL , 성공률 = '+str(score_n)+' , 전체 데이터 = '+str(total_m)+' , 성공 데이터 = '+str(suc_n)+' , 실패 데이터 = '+str(total_m-suc_n)+'\n'+'-'*30+'실패 이미지 리스트'+'-'*30+'\n')

            for z in fail_list_m:
                g.write(z + '\n')
            g.write('*' * 70 + 'error_list' + '*' * 70 + '\n')
            for z in error_list_m:
                g.write(z+'\n')
            g.write('*'*130+'\n')

            for z in fail_list_n:
                g_1.write(z + '\n')
            g_1.write('*' * 70 + 'error_list' + '*' * 70 + '\n')
            g.close()
            g_1.close()

        f.write('PROD = '+PROD+' , POINT = '+POINT+' , 성공률 = '+str(score)+' , 전체 데이터 = '+str(total)+' , 성공 데이터 = '+str(suc)+' , 실패 데이터 = '+str(total-suc)+'\n'+'-'*30+'실패 이미지 리스트'+'-'*30+'\n')
        f_1.write('PROD = '+PROD+' , POINT = '+POINT+' , 성공률 = '+str(score_o)+' , 전체 데이터 = '+str(total)+' , 성공 데이터 = '+str(suc_o)+' , 실패 데이터 = '+str(total-suc_o)+'\n'+'-'*30+'실패 이미지 리스트'+'-'*30+'\n')
        for z in fail_list:
            f.write(z+'\n')
        f.write('*'*70+'error_list'+'*'*70+'\n')
        for z in error_list:
            f.write(z+'\n')
        f.write('*'*130+'\n')
        for z in fail_list_o:
            f_1.write(z+'\n')
        f_1.write('*'*70+'error_list'+'*'*70+'\n')


        f.close()
        f_1.close()
print('end')
