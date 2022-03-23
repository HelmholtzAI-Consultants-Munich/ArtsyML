# ArtsyML: tf-packaging branch

## What is this?

This repository provides the code for a real time video stream which blends any people in the image with a biological image using style transfer. The result can look like the example below:

![Alt Text](https://github.com/HelmholtzAI-Consultants-Munich/ArtsyML/blob/remotes/origin/tf-packaging/demo.gif)


## tf-packaging branch

The version of code in this branch is made based on the "tf" branch in which the pytorch segmentation model was replaced by a tensorflow model in order to increase performance and subsequently to increase the framerate in video stream. The main change in "tf-packaging" branch in comparison with "tf" is that a new class "ArtsyML" is defined in order to store all models and apply the styling. The code is also minimized by removing all GUI tools since we are providing additional Flask-webpage to host the graphical interface for the user.

## Installation:

The code has been implemented using **Python 3.7+**. To install the necessary packages for this framework we would recommend the Python3 virtual environment. We first create a "venv" and name it ".venv_artsyml". Then we activate the environment and install the package using pip3:


```console
$ python3 -m venv .venv_artsyml
$ source .venv_artsyml/bin/activate
(.venv_artsyml) $ bash install_{cpu | gpu}.sh
```

Dependind on whether you have a GPU available or not, in the last line you should run ```install_gpu.sh``` or ```install_cpu.sh```

## How to run?

The ArtsyML package offers three options to users: running the style transfer live on a webcam output, running the style transfer on a single or multiple images and running the style transfer on a video. 

### Running on a webcam output
To start the video stream, simply open a terminal, activate the virtual environment type:

```console
(.venv_artsyml) $ python3 run_video_stream.py
```

To stop the video stream at any time the user can press the 'q' key on the keyboard. If no other argument is provided the script uses by default _pocimage.jpg_ as the style image. To provide a different style image, add it as an argument, e.g.:

```console
(.venv_artsyml) $ python run_video_stream.py --style_img tryout.jpg [rename these images]
```
### Running on images
To run ArtsyML on a single or multiple images,  simply open a terminal, activate the virtual environment type:

```console
(.venv_artsyml) $ python3 run_on_img.py --image your-image-path
```

The ```your-image-path``` variable can either be a directory where the images you want to transform are stored or the file path to a single image


### Running on a video
To run ArtsyML on a single or multiple images,  simply open a terminal, activate the virtual environment type:

```console
(.venv_artsyml) $ python3 run_on_img.py --video-file your-video-path
```
where ```your-video-path``` points to the file path of the video.

This will apply style transfer with ArtsyML on the entire video. If you wish to only apply the style transfer on a part of the video then type:
```console
(.venv_artsyml) $ python3 run_on_img.py --video-file your-video-path --start-frame 10 --end-frame 100
```
The example above will only apply the style transfer between frames 10 and 100 while leaving the rest of the frames untouched.

## API:
The API includes only one class, "ArtsyML". An instance of the class is initiated by a given style image and then segmentaiotn and styling combined model is stored as a method.

**artsyml.ArtsyML(style_image_file, which_seg_model='Deeplabv3', frame_shape = (720, 1280), seg_shape = (288,512,3)))**

* `style_image_file`: str

    Path to a style image.
    
* 'which_seg_model': str
    To chose a segmentation model to detect humans.

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

Similar steps are performed in all three script of this repository (```run_on_img.py```, ```run_on_video.py``` and ```run_video_stream.py```).Here we describe the process followed in ```run_video_stream.py```. In this script the webcam of the user is accessed and the webcam's frames are successively processed and displayed. The processing involves three steps:

1. The current frame is blended with a style image, using the approach suggested in [Neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer).
2. The current frame is segmented using either ```deeplabv3_resnet101``` or ```Mask-RCNN``` and the output masks is then converted to binary where only humans in the image are segmented. By default ```deeplabv3_resnet101``` is used for the ```run_video_stream.py``` script as it is much faster than ```Mask-RCNN```, whereas ```Mask-RCNN``` is used for ```run_on_img.py``` and ```run_on_video.py``` as it is generally more accurate. These configurations can be changed by changing the ```which_seg_model``` attribute of the ```ArtsyML``` class.
3. The outputs of the two previous steps are merged so that the output of the style net (step 1) is only aplied on the detected humans (from step 2). 

