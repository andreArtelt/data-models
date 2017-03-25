# German Traffic Sign classification
This folder contains **python code** for working with the [German Traffic Sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

## About
The **German Traffic Sign data set** is a single-image, multi-class classification problem.
It consists of more than **40 classes** and **50000 images**.

The task is to **predict** the correct **traffic sign** shown by an given **image**.
The raw images as well as some preprocessed features are available online.

You can **download the dataset** from [here](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip)
(Note this contains the raw images only!)

Each image is stored in **PPM format** and has a size between **15x15** and **250x250** pixels.


## Implementation
The "pipeline" works on **raw images**!
- **Preprocessing** (see parse.py): resizing (32x32), greyscaling and extracting the roi (*r*egion *o*f *i*nterest)
- **Model** (see model.py): Neural network (MLP or CNN) (Before learning pixels are scaled to [0,1]).

The data is **randomly split** into training (0.7), validation (0.1) test (0.2) set.
Depending on the split the **CNN** achives an **accuracy** of **0.994** on the test set.

### Run the code
 - Download the dataset (see above) and extract it into this folder (make sure the local and relative path to the first class is ``GTSRB/Final_Training/Images/00000/``).

 **Note:** On linux you can do this by simply executing ``sh download-data.sh``
 - Run ``python main.py``

### Requirements
- python 2.7
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [numpy](https://github.com/numpy/numpy)
- [keras](https://github.com/fchollet/keras) + backend
