import os
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
import argparse
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import time
from artsyml import ArtsyML

def get_args():
    parser = argparse.ArgumentParser(description='ArtsyML')
    parser.add_argument('--style_img', default='./images_style/style3.jpg')
    return parser.parse_args()

if __name__ == '__main__':
 
    args = get_args()  #"./images_style/style3.jpg"
    print("args",args)
    style_path = args.style_img
    _artsyml = ArtsyML(style_path)

    print('model loaded')
    
    cap = cv2.VideoCapture(0)
    
    first_time = time.time()
    prev_capture = time.time()
    avg_frame_rate = 0
    list_res = []
    
    while True:

        capture_time = time.time()
        print('Time between captures: ', capture_time - prev_capture)
        prev_capture = time.time()
        
        # Capture frame-by-frame
        success, frame = cap.read()

        cv2.imshow('Style Transfer', _artsyml.apply_style(frame).astype(np.uint8))
        #videoWriter.write(style_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()







