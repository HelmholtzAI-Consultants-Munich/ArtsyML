import tensorflow as tf
import tensorflow_hub as tfhub
from ._model import Deeplabv3
from ._utils import load_img, prepare_img_style
import os
import cv2
import numpy as np

class ArtsyML():
    def __init__(self, style_image_file, frame_shape = (720, 1280), seg_shape = (288,512,3)):
        self.frame_shape = frame_shape
        self.style_image_file = style_image_file
        self.seg_shape = seg_shape

        self.style_model = tfhub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')    
        self.style_image_abspath = os.path.abspath(style_image_file)
        self.read_style_image()

        self.seg_model = Deeplabv3(
            backbone = 'mobilenetv2', 
            input_shape = self.seg_shape, 
            OS=16
        )
        self.tf_combine_model = tf.function(self.combine)

    def read_style_image(self):
        _style_image = load_img(self.style_image_abspath)
        self.style_image = tf.stack([
                _style_image[:,:,:,2],
                _style_image[:,:,:,1],
                _style_image[:,:,:,0]
            ],
            axis = 3
        )

    def combine(self, content_image, seg_content_image):
        style_image_tensor = self.style_model(content_image, self.style_image)[0]
        seg_image_tensor = self.seg_model(seg_content_image) #**{'input.1': seg_content_image})[0]
        return style_image_tensor, seg_image_tensor
            
    def apply_style(self, frame):
        frame = frame.astype(np.uint8)
        frame_resized = cv2.resize(frame, (self.seg_shape[1], self.seg_shape[0]))
        seg_content_image = np.expand_dims(frame_resized / 127.5 - 1.,0)

        # Preparing the frame for the style net
        frame_tensor = tf.convert_to_tensor(frame_resized)
        content_image = prepare_img_style(frame_tensor) 

        style_image_tensor, seg_image_tensor = self.tf_combine_model(
            content_image = tf.constant(content_image),
            seg_content_image = tf.constant(seg_content_image)
        )

        style_image_tensor = tf.image.resize(style_image_tensor, (frame.shape[0], frame.shape[1])) 
        style_image_tensor = tf.squeeze(style_image_tensor, [0])*255 
        seg_image_tensor = tf.argmax(seg_image_tensor, axis=3)
        seg_mask = tf.math.equal(seg_image_tensor, 15) 
        seg_mask = tf.dtypes.cast(tf.stack([seg_mask] * 3, axis=-1), tf.float32)
        seg_mask = tf.image.resize(seg_mask, (frame.shape[0], frame.shape[1]))     
        seg_mask = tf.squeeze(seg_mask[0])

        #stylized_img = seg_mask*style_image_tensor+(1-seg_mask)*frame
        stylized_img = seg_mask*style_image_tensor+(1-seg_mask)*tf.cast(tf.convert_to_tensor(frame), tf.float32)
        stylized_img = stylized_img.numpy() 
        return stylized_img