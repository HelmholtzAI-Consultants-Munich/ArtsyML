import tensorflow as tf
print(tf.__version__)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import time
import functools






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
import PIL.Image
import time
import functools
import numpy as np
import cv2




# Function to load an image from a file, and add a batch dimension.
def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.io.decode_image(img, channels=3)
  

  return img


def preprocess_image_test(image, target_dim):
  # Resize the image so that the shorter dimension becomes 256px.
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = image[tf.newaxis, :]
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)

  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  return image


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


# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
  # Resize the image so that the shorter dimension becomes 256px.
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = image[tf.newaxis, :]
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)

  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  return image

# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image, style_predict_path):
  style_predictor = tf.lite.Interpreter(model_path=style_predict_path)
  style_predictor.allocate_tensors()
  # Set model input.
  input_details = style_predictor.get_input_details()

  style_predictor.set_tensor(input_details[0]["index"], preprocessed_style_image)

  
  style_predictor.invoke()

  # Calculate style bottleneck.
  style_bottleneck = style_predictor.tensor(
      style_predictor.get_output_details()[0]["index"]
      )()

  return style_bottleneck

# Run style transform on preprocessed style image
def run_style_transform(style_bottleneck, preprocessed_content_image, style_transform_path):

  style_transformer = tf.lite.Interpreter(model_path=style_transform_path)
  style_transformer.allocate_tensors()

  # Set model input.
  input_details = style_transformer.get_input_details()

  # Set model inputs.
  style_transformer.set_tensor(input_details[0]["index"], preprocessed_content_image)
  style_transformer.set_tensor(input_details[1]["index"], style_bottleneck)

  style_transformer.invoke()

  # Transform content image.
  stylized_image = style_transformer.tensor(
      style_transformer.get_output_details()[0]["index"]
      )()

  return stylized_image



def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return tensor #PIL.Image.fromarray(tensor)


# Load the models.
style_predict_path = tf.keras.utils.get_file('style_predict.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite')
style_transform_path = tf.keras.utils.get_file('style_transform.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite')




print('models_loaded')

content_blending_ratio = 0.0

style_path = 'pocimage.jpg'
style_image = load_img(style_path)
#style_image = tf.stack([style_image[:,:,:,2],style_image[:,:,:,1],style_image[:,:,:,0]],axis = 3)


# Calculate style bottleneck for the preprocessed style image.
style_bottleneck = run_style_predict(preprocess_image(style_image, 256), style_predict_path)


cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    style_bottleneck_content = run_style_predict(preprocess_image(frame, 256), style_predict_path)
    
    style_bottleneck_blended = content_blending_ratio * style_bottleneck_content + (1 - content_blending_ratio) * style_bottleneck

    # Stylize the content image using the style bottleneck.
    stylized_image = run_style_transform(style_bottleneck_blended, preprocess_image(frame, 384), style_transform_path)

    print(tf.shape(stylized_image))

    new_img = tensor_to_image(stylized_image)

    # Display the resulting frame
    cv2.imshow('Style Transfer',new_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()