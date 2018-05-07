README
======

This is an experimental attempt at detection of 'odd' behaviour in images.
The idea is that if we detect odd behaviour for an extended period (10-20 sec?)
that this may be indicative of a partial seizure.

We are using a neural network to determine whether an image is 'normal' or 'odd'
The method is based on the 'santa/no-santa' example at https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

The main change I have made is to increase the image size to 100x100 pixels, because any smaller than that and I could not detect Benjamin in the image, so didn't think the computer would manage.

INSTALLATION
============
This uses the python keras and tensorflow libraries.
A clean installation of Ubuntu 18.04 needs the following to get it working
(note that Ubuntu 18.04 comes with OpenCV V3.x, so worth using that LTS version now).
Note that I install mroe dependencies than you erally need below - some of it is my standard development set-up.

       sudo apt-get update
       sudo apt-get install build-essential git subversion mercurial cmake python-opencv python-opencv-apps 
       sudo apt-get install python-matplotlib python-scipy python-scikits-learn 
       sudo apt-get install python-httplib2
       sudo apt-get install python-pip
       sudo pip install imutils
       sudo pip install keras
       sudo apt-get install emacs25
       sudo pip install tensorflow
       pip install bottle
       mkdir OpenSeizureDetector
       cd OpenSeizureDetector/
       git clone https://github.com/OpenSeizureDetector/VideoBehaviourDetector.git
       cd VideoBehaviourDetector/


Training the Network
====================
But first we have to train the neural network with example images.....

Proportion of negative (=normal) samples compared to positive (odd) ones
should be 16 to 18% (ie a lot less normal ones than odd):  https://pdfs.semanticscholar.org/235d/cf221298ae74252b10ef3f69acd4f7e1585f.pdf

But I have ignored this and used two normal for each odd image below.....

Training Results, 29apr2018
===========================
Used 784 'Odd' images, and 2x784 'Normal' ones to train the network.
Then passed all of my available categorised images (over 13000) through the
analysis - it took 6 min on my Core I7 laptop.
This gave an 'odd' detection reliability of 96% and a false positive rate of 6%.

The odd detection failures are subtle ones - lying on front proped up on one arm etc.   I think the false positives are mostly lying on back on pillows, which does not look too different to kneeling up in a similar position.

Progress!
************************
*     RESULTS          *
************************

Number of Odd Images Tested = 784
Number of Odd Image Errors  = 38
  Odd Detection Reliability = 96%

Number of Normal Images Tested = 12547
Number of Normal Image Errors  = 870
  Normal Detection Reliability = 94%

