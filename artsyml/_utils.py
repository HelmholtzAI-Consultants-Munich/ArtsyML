import tensorflow as tf
import numpy as np
import sys
# fix this import
sys.path.insert(0, '../models/research')
from object_detection.utils import ops as utils_ops

def convert_to_mask(mask, bbox, frame_shape):
    mask = tf.convert_to_tensor(mask)
    bbox = tf.convert_to_tensor(bbox)
    mask = tf.expand_dims(mask, axis=0)
    bbox = tf.expand_dims(bbox, axis=0)
    mask_reframed = utils_ops.reframe_box_masks_to_image_masks(mask, bbox, frame_shape[0], frame_shape[1])
    mask_reframed = tf.cast(mask_reframed > 0.5, tf.uint8)
    mask = mask_reframed.numpy()
    return mask[0]

def get_nearest_person_seg(result, frame_shape):
    boxes_to_keep = []
    detection_classes_to_keep = []
    ids_to_keep = []
    for it in range(result['detection_boxes'][0].shape[0]):
        if result['detection_scores'][0][it] > 0.3:
            boxes_to_keep.append(list(result['detection_boxes'][0][it]))
            detection_classes_to_keep.append(int(result['detection_classes'][0][it]))
            ids_to_keep.append(it)
    num_people = detection_classes_to_keep.count(1)
    if num_people==0:
        mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    elif num_people==1:
        person_id = detection_classes_to_keep.index(1)
        mask = result['detection_masks'][0][ids_to_keep[person_id]]
        bbox = result['detection_boxes'][0][ids_to_keep[person_id]]
        mask = convert_to_mask(mask, bbox, frame_shape)
    else:
        central_person_id = 0
        max_box_size = 0
        people_ids = [pid for pid, pc in enumerate(detection_classes_to_keep) if pc==1]
        for pid in people_ids: 
            box_size = (boxes_to_keep[pid][2]-boxes_to_keep[pid][0])*(boxes_to_keep[pid][3]-boxes_to_keep[pid][1])
            if box_size > max_box_size: 
                central_person_id = ids_to_keep[pid]
                max_box_size = box_size
        mask = result['detection_masks'][0][central_person_id]
        bbox = result['detection_boxes'][0][central_person_id]
        mask = convert_to_mask(mask, bbox, frame_shape)
    mask = np.expand_dims(mask, axis=2)
    return mask

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
