import os
import argparse
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import functools
import time

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def prepare_img_style(img):
    max_dim = 512
    img = tf.convert_to_tensor(img)
    #img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def prepare_img_seg(img):
    seg_content_image = np.expand_dims(img / 127.5 - 1., 0)
    seg_content_image = tf.transpose(seg_content_image, perm=[0, 3, 1, 2])
    seg_content_image = tf.convert_to_tensor(seg_content_image)
    seg_content_image = tf.image.convert_image_dtype(seg_content_image, tf.float32)
    return seg_content_image

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor 

def combine(content_image,style_image,seg_content_image):
    style_image_tensor = style_model(content_image, style_image)[0]
    res = seg_model(**{'input.1': seg_content_image})[0]
    return style_image_tensor,res
    

def get_args():
    parser = argparse.ArgumentParser(description='ArtsyML')
    parser.add_argument('--style_img', default='./images_style/style3.jpg')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    style_path = args.style_img
    style_image = load_img(style_path)
    
    style_image = tf.stack([style_image[:,:,:,2],style_image[:,:,:,1],style_image[:,:,:,0]],axis = 3)
    style_model = tfhub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    seg_model = tf.saved_model.load('tf_deeplabv3.pb') #Deeplabv3(backbone='mobilenetv2',input_shape=(288,512,3), OS=16)
    
    tf_combine_model = tf.function(combine)

    print('model loaded')
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter('video.avi', fourcc, 30.0, (288,512))
    prev_capture = time.time()
    
    while True:
        
        capture_time = time.time()
        print('Time between captures: ', capture_time - prev_capture)
        prev_capture = capture_time

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Preparing the frame for the style net
        content_image = prepare_img_style(frame)
        frame = cv2.resize(frame, (content_image.shape[2], content_image.shape[1]))
        seg_content_image = prepare_img_seg(frame)

        style_image_tensor,seg_image_tensor = tf_combine_model(tf.constant(content_image),tf.constant(style_image),tf.constant(seg_content_image))

        style_img = tensor_to_image(style_image_tensor)
        seg_mask = np.array(tf.argmax(seg_image_tensor, axis=1)[0,:,:]) # model returns a list ['out', 'aux'], 'aux' only used during training
        seg_mask[seg_mask!=15] = 0
        seg_mask[seg_mask==15] = 1
    
        # keep people only from style image and background only from original frame
        style_img =  (1-seg_mask[:,:,None])*frame + seg_mask[:,:,None]*style_img
        style_img = style_img.astype(np.uint8)
        # Display the resulting frame
        cv2.imshow('Style Transfer', style_img)
        videoWriter.write(style_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()







