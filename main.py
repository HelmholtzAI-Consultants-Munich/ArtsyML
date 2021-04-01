# -*- coding: utf-8 -*-

__author__      = "Saulius Lukse"
__copyright__   = "Copyright 2016, kurokesu.com"
__version__ = "0.1"
__license__ = "GPL"


from PyQt5 import QtCore, QtGui, uic, QtWidgets
import sys
import cv2
import numpy as np
import threading
import time
import queue

import os
import argparse
import tensorflow as tf
import tensorflow_hub as tfhub
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import functools
import cv2
import time

import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
from PIL import Image

running = False
capture_thread = None
form_class = uic.loadUiType("simple.ui")[0]
q = queue.Queue()
 
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

def grab(cam, queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    while(True):
        while(running):
            frame = {}        
            capture.grab()
            retval, img = capture.retrieve(0)
            frame["img"] = img

            if queue.qsize() < 10:
                queue.put(frame)
            else:
                print(queue.qsize())

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

class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()



class MyWindowClass(QtWidgets.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.startButton.clicked.connect(self.start_clicked)
        self.screenshotButton.setStyleSheet("border-image : url(./images_gui/imagesRound.jpg);")
        self.screenshotButton.clicked.connect(self.takeScreenshot)
        self.screenshotButton.hide()

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)       

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.styleImg = load_img('./images_style/style3.jpg') # default style
        self.styleImg = tf.stack([self.styleImg[:,:,:,2],self.styleImg[:,:,:,1],self.styleImg[:,:,:,0]],axis = 3)
        
        self.styleImgButton1.clicked.connect(self.setStyle1)
        self.styleImgButton1.setStyleSheet("border-image : url(./images_style/style1.jpg);")
        
        self.styleImgButton2.clicked.connect(self.setStyle2)
        self.styleImgButton2.setStyleSheet("border-image : url(./images_style/style2.jpg);")

        self.styleImgButton3.clicked.connect(self.setStyle3)
        self.styleImgButton3.setStyleSheet("border-image : url(./images_style/style3.jpg);")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('DEVICE', self.device )
        self.seg_model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
        self.seg_model = self.seg_model.to(device=self.device)
        self.seg_model.eval()

        self. style_model = tfhub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def takeScreenshot(self):
        mpl.image.imsave('screenshot.png', self.img)
    
    def setStyle1(self):
        self.styleImg = load_img('./images_style/style1.jpg')
        self.styleImg = tf.stack([self.styleImg[:,:,:,2],self.styleImg[:,:,:,1],self.styleImg[:,:,:,0]],axis = 3)

    def setStyle2(self):
        self.styleImg = load_img('./images_style/style2.jpg')
        self.styleImg = tf.stack([self.styleImg[:,:,:,2],self.styleImg[:,:,:,1],self.styleImg[:,:,:,0]],axis = 3)

    def setStyle3(self):
        self.styleImg = load_img('./images_style/style3.jpg')
        self.styleImg = tf.stack([self.styleImg[:,:,:,2],self.styleImg[:,:,:,1],self.styleImg[:,:,:,0]],axis = 3)

    def start_clicked(self):
        global running
        if self.startButton.text() == 'Start video':
            running = True
            self.screenshotButton.show()
            if capture_thread.is_alive():
                self.screenshotButton.show()
            else:
                capture_thread.start()
            self.startButton.setEnabled(False)
            self.startButton.setText('Starting...')
        elif self.startButton.text() == 'Stop video':
            running = False
            self.screenshotButton.hide()
            self.ImgWidget.hide()
            self.startButton.setText('Start video')

    def get_stylized(self, frame):
        # Preparing the frame for the style net
        content_image = prepare_img(frame)
        style_img = self.style_model(tf.constant(content_image), tf.constant(self.styleImg))[0]
        style_img = tensor_to_image(style_img)
        
        # Preparing the frame for the segmentation net 
        # resize to same shape as output of style net
        frame = cv2.resize(frame, (style_img.shape[1], style_img.shape[0]))
        input_tensor = self.preprocess(frame)
        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device=self.device)
        
        with torch.no_grad():
            seg_output = self.seg_model(input_batch)['out'][0]
            seg_output = seg_output.detach().argmax(0)

        # edit segmentation mask to binary to keep people only
        seg_mask = seg_output.cpu().data.numpy() 
        seg_mask[seg_mask!=15] = 0
        seg_mask[seg_mask==15] = 1

        # keep people only from style image and background only from original frame
        style_img =  (1-seg_mask[:,:,None])*frame + seg_mask[:,:,None]*style_img
        style_img = style_img.astype(np.uint8)
        return style_img

    def update_frame(self):
        if not q.empty():
            self.startButton.setEnabled(True)
            self.startButton.setText('Stop video')
            frame = q.get()
            img = frame["img"]
            # style transfer and segmentation
            self.img = self.get_stylized(img)

            img_height, img_width, img_colors = self.img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            img = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)

    def closeEvent(self, event):
        global running
        running = False



capture_thread = threading.Thread(target=grab, args = (0, q, 1920, 1080, 30))

app = QtWidgets.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('Artsy ML')
w.show()
app.exec_()
