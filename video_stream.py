
import os
import tensorflow as tf
import tensorflow_hub as hub
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import time
import functools
import numpy as np
import cv2

import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
from PIL import Image

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

def prepare_img(img):
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

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return tensor #PIL.Image.fromarray(tensor)


if __name__ == '__main__':

  style_path = 'pocimage.jpg'
  style_image = load_img(style_path)
  style_image = tf.stack([style_image[:,:,:,2],style_image[:,:,:,1],style_image[:,:,:,0]],axis = 3)

  style_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
  seg_model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
  seg_model.eval()

  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  print('loaded model')

  cap = cv2.VideoCapture(0)
  while(True):
      # Capture frame-by-frame
      ret, frame = cap.read()
      
      # Preparing the frame for the style net
      content_image = prepare_img(frame)
      style_image_tensor = style_model(tf.constant(content_image), tf.constant(style_image))[0]
      style_img = tensor_to_image(style_image_tensor)

      # Preparing the frame for the segmentation net 
      # resize to same shape as output of style net
      frame = cv2.resize(frame, (style_img.shape[1], style_img.shape[0]))
      input_tensor = preprocess(frame)
      # create a mini-batch as expected by the model
      input_batch = input_tensor.unsqueeze(0) 
      
      with torch.no_grad():
          seg_output = seg_model(input_batch)['out'][0]
          seg_output_predictions = seg_output.argmax(0)

      # edit segmentation mask to binary to keep people only
      seg_mask =  seg_output_predictions.cpu().numpy()
      seg_mask[seg_mask!=15] = 0
      seg_mask[seg_mask==15] = 1

      # keep people only from style image and background only from original frame
      style_img =  (1-seg_mask[:,:,None])*frame + seg_mask[:,:,None]*style_img
      style_img = style_img.astype(np.uint8)

      # Display the resulting frame
      cv2.imshow('Style Transfer',style_img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()