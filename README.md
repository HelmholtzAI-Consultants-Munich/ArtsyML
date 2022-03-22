# ArtsyML: tf-packaging branch

## What is this?

This repository provides the code for a real time video stream which blends any people in the image with a biological image. The result can look sotheing like the example below:

[add a GIF here](https://github.com/HelmholtzAI-Consultants-Munich/ArtsyML/blob/remotes/origin/tf-packaging/demo.gif)


## tf-packaging branch

The version of code in this branch is made based on the "tf" branch in which the pytorch segmentation model was replaced by a tensorflow model in order to increase performance and subsequently to increase the framerate in video stream. The main change in "tf-packaging" branch in comparison with "tf" is that a new class "ArtsyML" is defined in order to store all models and apply the styling. The code is also minimized by removing all GUI tools since we are providing additional Flask-webpage to host the graphical interface for the user.

## Installation:

The code has been implemented using **Python 3.7+**. To install the necessary packages for this framework we would recommend the Python3 virtual environment. We first create a "venv" and name it ".venv_artsyml". Then we activate the environment and install the package using pip3:


```console
$ python3 -m venv .venv_artsyml
$ source .venv_artsyml/bin/activate(.venv_artsyml) 
(.venv_artsyml) $ pip3 install -e .
```

To install create a new conda environment (has been tested with **Python 3.7, 3.9.+**) and from within it run: ```bash install.sh```

## How to run?
To start the video stream, simply open a terminal, activate the virtual environment type:

```console
(.venv_artsyml) $ python3 video_stream.py
```

To stop the video stream at any time the user can press the 'q' key on the keyboard. If no other argument is provided the script uses by default _pocimage.jpg_ as the style image. To provide a different style image, add it as an argument, e.g.:


```console
(.venv_artsyml) $ python video_stream.py --style_img tryout.jpg [rename these images]
```
## API:
The API includes only one class, "ArtsyML". An instance of the class is initiated by a given style image and then segmentaiotn and styling combined model is stored as a method.

**artsyml.ArtsyML(style_image_file, frame_shape = (720, 1280), seg_shape = (288,512,3)))**

* `style_image_file`: str

    Path to a style image.

* `frame_shape`: tuple

    Shape of the output frame.

* `seg_shape`: tuple

    Shape of the segmentation model.

### artsyml.ArtsyML atrributes and methods:

* `artsyml.ArtsyML.style_image_abspath`

    Gives the absolute path to the style image.

* `artsyml.ArtsyML.create_model(style_image_file: str)`

    Creates a combined model for segmentation and styling for a given style image `style_image_file`. The method is internally called during initiation of an instance. Thus it is only needed to be called for changing the style image.

* `artsyml.ArtsyML.apply_style(frame: numpy.ndarray)`

    Gets a frame as an input image with type of numpy.ndarray returns a styled frame with type of numpy.ndarray.




## How does this work?

The main script of this repository is ```video_stream.py```. In this script the webcam of the user is accessed and the webacam's frames are successively processed and displayed. The processing involves three steps:

1. The current frame is blended with a style image, using the approach suggested in [Neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer).
2. The current frame is segmented using ```deeplabv3_resnet101``` and the output masks is then converted to binary where only humans in the image are segmented.
3. The outputs of the two previous steps are merged so that the output of the style net (step 1) is only aplied on the detected humans (from step 2). 

