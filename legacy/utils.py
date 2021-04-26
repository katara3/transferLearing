import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random, os


def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)

        
def draw_multi_img(args,img_path,sample_ct,rows,cols):
    x1=args['x1']
    x2=args['x2']
    y1=args['y1']
    y2=args['y2']

    idx=1
    axes=[]
    fig = plt.figure(figsize=[14,7]) 
    if sample_ct < len(os.listdir(img_path)):
        img_list = random.sample(os.listdir(img_path),sample_ct)
    else:
        img_list = os.listdir(img_path)[:sample_ct]

    for img_name in img_list:
        if idx > rows*cols:
            break
        tmp = Image.open(os.path.join(img_path,img_name))
        drawing = ImageDraw.Draw(tmp)
        draw_rectangle(drawing, ((x1,y1),(x2,y2)), color="green", width=6)

        axes.append(fig.add_subplot(rows,cols,idx))
        axes[-1].set_title(str(idx)+': '+img_name[-10:])
        axes[-1].axis('off')
        plt.imshow(tmp)
        idx+=1

    fig.tight_layout()
    plt.show()


def get_ng_case_ct(args):
    ng_img_path=os.path.join(args['prod'],args['position'],'NG')
    ng_img_list=os.listdir(ng_img_path)
    if len(ng_img_list) > 0:
        if os.path.isfile(os.path.join(ng_img_path,ng_img_list[0])):
            ng_case_ct=1
        else:
            ng_case_ct=len(ng_img_list)
    else:
        ng_case_ct=0
    
    return ng_case_ct


def draw_single_img(args,img_path):
    tmp = Image.open(os.path.join(img_path,random.sample(os.listdir(img_path),1)[0]))
    x1=args['x1']
    x2=args['x2']
    y1=args['y1']
    y2=args['y2']

    args['width']=x2-x1
    args['height']=y2-y1


    drawing = ImageDraw.Draw(tmp)
    draw_rectangle(drawing, ((x1,y1),(x2,y2)), color="green", width=6)
    plt.imshow(tmp)
