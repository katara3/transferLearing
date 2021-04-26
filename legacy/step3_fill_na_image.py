import os
import sys
import random
import shutil
import argparse
from PIL import Image, ImageDraw


parser = argparse.ArgumentParser(description='현재 포지션의 NG이미지가 없는경우 다른 제품의 포지션 이미지로 채운다.')

parser.add_argument('--path',
                    action='store',
                    required=True,
                    help = '복사를 수행할 다른 제품의 포지션 이미지 경로를 입력합니다.',
                    dest = 'from_img_path')
parser.add_argument('--prod',
                    action='store',
                    help='학습 대상 제품을 입력합니다. (EX: 06_C300)',
                    required=True,
                    dest = 'prod')
parser.add_argument('--position',
                    action='store',
                    help='학슬 대상 포지션을 입력합니다. (EX: 1)',
                    required=True,
                    dest = 'position')
parser.add_argument('--train_img_ct',
                    action='store',
                    required=True,
                    help='학습 이미지 갯수를 입력합니다. (EX: 2000)',
                    dest = 'train_img_ct')
parser.add_argument('--valid_img_ct',
                    action='store',
                    required=True,
                    help='검증 이미지 갯수를 입력합니다. (EX: 200)',
                    dest = 'valid_img_ct')

if len(sys.argv) == 1:
    parser.print_help()

args = parser.parse_args()
from_img_path = args.from_img_path
train_img_ct = int(args.train_img_ct)
valid_img_ct = int(args.valid_img_ct)
position = args.position
prod =  args.prod
from_img_list = os.listdir(from_img_path)

os.makedirs(os.path.join(prod,'train',position+'_ng'), exist_ok=True)
os.makedirs(os.path.join(prod,'valid',position+'_ng'), exist_ok=True)

x1 = 100
x2 = 300
y1 = 100
y2 = 300
ng_img_ct=0
for img_name in random.sample(from_img_list,train_img_ct):
    tmp = Image.open(os.path.join(from_img_path,img_name))
    tmp = tmp.crop((x1,y1,x2,y2))
    tmp.save(os.path.join(os.path.join(prod,'train',position+'_ng'),'{0:06d}.png'.format(ng_img_ct)))
    ng_img_ct+=1

for img_name in random.sample(from_img_list,valid_img_ct):
    tmp = Image.open(os.path.join(from_img_path,img_name))
    tmp = tmp.crop((x1,y1,x2,y2))
    tmp.save(os.path.join(os.path.join(prod,'valid',position+'_ng'),'{0:06d}.png'.format(ng_img_ct)))
    ng_img_ct+=1
    
print("*"*50)
print("finish")
