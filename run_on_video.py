import os
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
import argparse
import time
from artsyml import ArtsyML


def get_args():
    parser = argparse.ArgumentParser(description='ArtsyML')
    parser.add_argument('--video-file')
    parser.add_argument('--style-img', default='./images_style/style3.jpg')
    parser.add_argument('--start-frame', default=0, type=int)
    parser.add_argument('--end-frame', default=-1, type=int)
    parser.add_argument('--save-file', default='video_style.mp4')
    return parser.parse_args()

if __name__ == '__main__':
 
    args = get_args()  
    print("args",args)
    style_path = args.style_img
    video_dir_path = args.video_file

    _artsyml = ArtsyML(style_path, which_seg_model='Mask-RCNN')
    print('model loaded')

    # open video
    cap = cv2.VideoCapture(video_dir_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Video fps: ', fps,' height: ', height, 'width: ', width)
    if not args.save_file.endswith('.mp4'):
        print('Specified output file must be in mp4 format - will save output as video_style.mp4')
        output_path = 'video_style.mp4'
    else: output_path = args.save_file
    out = cv2.VideoWriter(output_path,
                        cv2.VideoWriter_fourcc(*'MP4V'), 
                        fps,
                        (width, height))
    end_frame = args.end_frame
    if end_frame == -1: end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # loop through video
    frame_id = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            if frame_id >= args.start_frame and frame_id < end_frame:
                frame = _artsyml.apply_style(frame).astype(np.uint8)
            out.write(frame)
            frame_id+=1
        else: break
        
    cap.release()
    out.release()








