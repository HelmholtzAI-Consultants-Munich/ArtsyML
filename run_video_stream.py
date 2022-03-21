import os
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
import argparse
import time
from artsyml import ArtsyML

def get_args():
    parser = argparse.ArgumentParser(description='ArtsyML')
    parser.add_argument('--style_img', default='./images_style/style3.jpg')
    return parser.parse_args()

if __name__ == '__main__':
 
    args = get_args() 
    print("args",args)
    style_path = args.style_img
    _artsyml = ArtsyML(style_path) #, which_seg_model='Mask-RCNN'
    print('model loaded')
    
    cap = cv2.VideoCapture(0)
        
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if success:
            cv2.imshow('ArstyML', _artsyml.apply_style(frame).astype(np.uint8))
            #videoWriter.write(style_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()







