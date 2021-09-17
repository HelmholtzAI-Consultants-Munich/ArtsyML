# ArtsyML: tf-packaging branch

## What is this?

This repository provides the code for a real time video stream which blends any people in the image with a biological image. The result can look sotheing like the example below:

[add a GIF here] e.g. [Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)


## tf-packaging branch

The version of code in this branch is made based on the "tf" branch in which the pytorch segmentation model was replaced by a tensorflow model in order to increase performance and subsequently to increase the framerate in video stream. The main change in "tf-packaging" branch in comparison with "tf" is that a new class "ArtsyML" is defined in order to store all models and apply the styling.

## Installation:

The code has been implemented using **Python 3.7+**. To install the necessary packages for this framework we would recommend the Python3 virtual environment. We first create a "venv" and name it ".venv_artsyml". Then we activate the environment and install the package using pip3:


```console
$ python3 -m venv .venv_artsyml
$ source .venv_artsyml/bin/activate(.venv_artsyml) 
$ pip3 install -e .
```

## How to run?
To start the video stream, simply open a terminal, activate the virtual environment type:

```console
(.venv_artsyml) $ python3 video_stream.py
```

To stop the video stream at any time the user can press the 'q' key on the keyboard. If no other argument is provided the script uses by default _pocimage.jpg_ as the style image. To provide a different style image, add it as an argument, e.g.:


```console
python video_stream.py --style_img tryout.jpg [rename these images]
```

## How does this work?

The main script of this repository is ```video_stream.py```. In this script the webcam of the user is accessed and the webacam's frames are successively processed and displayed. The processing involves three steps:

1. The current frame is blended with a style image, using the approach suggested in [Neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer).
2. The current frame is segmented using ```deeplabv3_resnet101``` and the output masks is then converted to binary where only humans in the image are segmented.
3. The outputs of the two previous steps are merged so that the output of the style net (step 1) is only aplied on the detected humans (from step 2). 

