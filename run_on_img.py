import os
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
import argparse
import time
from artsyml import ArtsyML


def get_args():
    parser = argparse.ArgumentParser(description='ArtsyML')
    parser.add_argument('--image')
    parser.add_argument('--output', default='artsy_results', help='Specify directory where to save results')
    parser.add_argument('--style_img', default='./images_style/style3.jpg')
    return parser.parse_args()

if __name__ == '__main__':
 
    args = get_args() 
    print("args",args)
    style_path = args.style_img
    img_dir_path = args.image

    if os.path.isdir(img_dir_path):
        list_images = [os.path.join(img_dir_path, file) for file in os.listdir(img_dir_path) if (file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg'))]
    else:
        list_images = [img_dir_path]

    if not os.path.exists(args.output): os.mkdir(args.output)
    
    _artsyml = ArtsyML(style_path, which_seg_model='Mask-RCNN')
    print('model loaded')
    
    for img_path in list_images:
        img_file = os.path.split(img_path)[-1]
        img = cv2.imread(img_path).astype(np.uint8)
        result = _artsyml.apply_style(img).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, img_file), result.astype(np.uint8))
    print('Done - results stored in: ', args.output)





