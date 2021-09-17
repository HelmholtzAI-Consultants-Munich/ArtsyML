
import tensorflow as tf
import numpy as np

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


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    #if np.ndim(tensor)>3:
        #assert tensor.shape[0] == 1
    tensor = tensor[0]
    return tensor 
