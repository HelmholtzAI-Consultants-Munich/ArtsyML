import os
import argparse
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import time
from model import Deeplabv3

def prepare_img_style(img, max_dim=512): #1280

    img = tf.image.convert_image_dtype(img, tf.float32)
    #shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    #long_dim = max(shape)
    #scale = max_dim / long_dim
    #new_shape = tf.cast(shape * scale, tf.int32)
    #img = tf.image.resize(img, [288,512])
    img = img[tf.newaxis, :]
    return img


def prepare_img_seg(img):
    seg_content_image = np.expand_dims(img / 127.5 - 1., 0)
    seg_content_image = tf.transpose(seg_content_image, perm=[0, 3, 1, 2])
    seg_content_image = tf.convert_to_tensor(seg_content_image)
    seg_content_image = tf.image.convert_image_dtype(seg_content_image, tf.float32)
    return seg_content_image

def load_img(path_to_img, max_dim=512): #1024):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    #img = prepare_img_style(img, max_dim)
    
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    
    return img


# def tensor_to_image(tensor):
#     tensor = tensor*255
#     tensor = np.array(tensor, dtype=np.uint8)
#     #if np.ndim(tensor)>3:
#         #assert tensor.shape[0] == 1
#     tensor = tensor[0]
#     return tensor 

def combine(content_image,style_image,seg_content_image):
    style_image_tensor = style_model(content_image, style_image)[0]
    segmetation_mask = seg_model(seg_content_image)
    return style_image_tensor,segmetation_mask
    

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
    
    seg_sizes = (288,512,3)
    seg_model = Deeplabv3(backbone='mobilenetv2', input_shape=seg_sizes, OS=16)
    
    tf_combine_model = tf.function(combine)
    print('model loaded')
    
    # load video to simulate cam reading
    arr = np.load('./input_videos/long_moving_720p.npy') 
    
    frame_dim1, frame_dim2 = (arr[0].shape[0], arr[0].shape[1])  #(360, 640) #(720, 1280) #(1080, 1920)
    
    first_time = time.time()
    prev_capture = time.time()
    avg_frame_rate = 0
    list_res = []
    
    for i in range(arr.shape[0]):
        
        if i>1:
            avg_frame_rate += time.time() - prev_capture 
        prev_capture = time.time()

        # Capture frame-by-frame
        frame = arr[i].astype(np.uint8)
        frame_t = tf.cast(tf.convert_to_tensor(frame), tf.float32)


        frame_resized = cv2.resize(frame, (seg_sizes[1], seg_sizes[0]))
        seg_content_image = np.expand_dims(frame_resized / 127.5 - 1.,0)

        # Preparing the frame for the style net
        frame_tensor = tf.convert_to_tensor(frame_resized)
        content_image = prepare_img_style(frame_tensor) #, frame_dim2)

        tik = time.time()
        #print('preprocessing time: ',tik-prev_capture)

        style_image_tensor,seg_image_tensor = tf_combine_model(tf.constant(content_image),tf.constant(style_image),tf.constant(seg_content_image))

        tok = time.time()
        print('inference time: ',tok-tik)

        style_image_tensor = tf.image.resize(style_image_tensor, (frame.shape[0], frame.shape[1])) 
        style_image_tensor = tf.squeeze(style_image_tensor, [0])*255 
        seg_image_tensor = tf.argmax(seg_image_tensor, axis=3)
        seg_mask = tf.math.equal(seg_image_tensor,15) 
        seg_mask = tf.dtypes.cast(tf.stack([seg_mask] * 3, axis=-1), tf.float32)
        seg_mask = tf.image.resize(seg_mask, (frame.shape[0], frame.shape[1]))     
        seg_mask = tf.squeeze(seg_mask[0])

        #stylized_img = seg_mask*style_image_tensor+(1-seg_mask)*frame
        stylized_img = seg_mask*style_image_tensor+(1-seg_mask)*tf.cast(tf.convert_to_tensor(frame), tf.float32)
        stylized_img = stylized_img.numpy() #tf.make_ndarray(stylized_img) # np.array(stylized_img)
        list_res.append(stylized_img.astype(np.uint8))
        tik = time.time()
        print('postprocessing time: ',tik-tok)
        print(' ')
        

    fps = round((arr.shape[0]-2)/avg_frame_rate)
    print('Average SPF: ', avg_frame_rate/(arr.shape[0]-2),'and FPS: ', fps)
    
    # save result as video
    style_name = style_path.split('/')[-1].split('.')[0]
    output_name = ('_').join([style_name, str(frame_dim1)+'p', str(fps)+'fps.mp4'])
    
    if fps==16: # for some reason i get an error when saving video with 16fps
        print('OhOh')
        fps=17 

    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_dim2, frame_dim1)) 
    for stylized_img in list_res:
        out.write(stylized_img)
    out.release()