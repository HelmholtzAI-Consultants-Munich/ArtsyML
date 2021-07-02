# ArtsyML

## What is this?
This repository provides the code for a real time video stream which blends any people in the image with a biological image. The result can look sotheing like the example below:

[add a GIF here] e.g. [Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)

## Installation:
The code has been implemented using **Python 3.7**. To install the necessary packages for this framework with conda run:
```
conda env create -f environment.yaml
```

## How to run?

## How to run?

To start the video stream, simply open a terminal and type:
```
python video_stream.py
```

To stop the video stream at any time the user can press the 'q' key on the keyboard. If no other argument is provided the script uses by default _pocimage.jpg_ as the style image. To provide a different style image, add it as an argument, e.g.:

```
python video_stream.py --style_img tryout.jpg [rename these images]
```

## How does this work?

The main script of this repository is ```video_stream.py```. In this script the webcam of the user is accessed and the webacam's frames are successively processed and displayed. The processing involves three steps:

1. The current frame is blended with a style image, using the approach suggested in [Neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer).
2. The current frame is segmented using ```deeplabv3_resnet101``` and the output masks is then converted to binary where only humans in the image are segmented.
3. The outputs of the two previous steps are merged so that the output of the style net (step 1) is only aplied on the detected humans (from step 2). 

