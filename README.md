# Computer vision (ENGN4528) project repository

This directory holds Sam Toyer (u5568237) and Kuangyi Xing's (u5817313) ENGN4528
term project on face detection and recognition.

## Running our code

Below is a list of steps required to install the
dependencies for our project and run some brief demos. Note that our deep
learning dependencies (Keras, Tensorflow, etc.) are difficult to get running on
Windows, so you may not be able to complete these steps unless you use a Unix
machine (preferably Ubuntu 17.04, which is what we used). Assuming you have an
appropriate environment, the complete list of steps is as follows:

1. Download Anaconda for Python 3.5+ for your system
   from [continuum.io](https://www.continuum.io/DOWNLOADS).
2. In a shell, type
   `conda create -n engn4528-toyer-xing --file dependencies/conda-requirements.txt`.
   This will create a new Anaconda/Python
   3 environment called "engn4528-toyer-xing" which includes all the necessary
   dependencies.
3. Use `source activate engn4528-toyer-xing` from a shell to enter the newly
   created environment. You'll be able to tell that your in the environment
   because your shell prompt will have `(engn4528-toyer-xing)` prepended to it.
4. Run `pip install -r dependencies/pip-requirements.txt`. This will install the
   remaining dependencies for our project. At this point, you may wish to
   install a GPU copy of Tensorflow if you have an NVIDIA GPU; the default CPU
   is far slower than even an embedded NVIDIA GPU (we tested on a laptop's
   NVIDIA 940MX)
5. Now you can run `python3 demo.py`. This script is meant to do a few things:

    1. It downloads our trained models and data from
       [users.cecs.anu.edu.au](http://users.cecs.anu.edu.au/~u5568237/engn4528/engn4528-data.zip).
       This file is very large (>200MB), and was not practical to upload to
       Wattle.
    2. It presents you with options to choose one of three demos, each of which
       runs our detector on an ANU group photo (**not included due to privacy
       concerns**), then applies one of our recognisers (Eigenfaces, VAE, or
       IDVAE, depending on the selected option).
    3. It runs the demo you selected. The demo will show you three windows: a
       group photo and two (initially blank) detail windows. You may have to
       switch between the windows it creates to find the group photo. If you
       click on a detected face in the group photo and wait a few seconds, the
       detail windows will show you the cropped face, a recovered version of the
       face, and the nearest matches to that person in the ANU Faces dataset. It
       may take a few seconds to complete after a detection is selected,
       depending on how powerful your machine is.

   A couple of words on performance: firstly, the VAE and IDVAE will have to run
   on a CUDA-capable GPU (a laptop GPU is fine) to finish within the 20s time
   limit. Secondly, even on a powerful machine `demo.py` will take much more
   than 20s, as it must (a) download data, and (b) must create descriptors for
   the 414 students in the ANU Faces dataset (not just the ones in the training
   photo---these descriptors *could* be saved for reuse, but we have not
   implemented that functionality); nevertheless, the detection and recognition
   pipeline itself should still take well under 20s.

## Code structure

The `fidentify/` subdirectory is a Python package containing most of the code
for this project. This directory (the root) also contains a few scripts for
interacting with our system. The following files will likely be of interest:

- `demo.py`: Runs a complete demonstration on pre-trained models. This is the
  easiest way to see how our system works.
- `test_on_group_photo.py`: Coordinates detection, alignment and recognition
  code to apply the system to a group photo. `demo.py` will run this for you.
- `train_vae.py`: Responsible for training the VAE and IDVAE. Shows training
  protocol used for experiments in report.
- `fidentify/detector.py`: Face detection and alignment code.
- `fidentify/eigenfaces.py`: Our implementation of Eigenfaces.
- `fidentify/vae.py`: VAE and IDVAE code, including model definitions.
