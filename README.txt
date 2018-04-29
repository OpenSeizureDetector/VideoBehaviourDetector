README
======

This is an experimental attempt at detection of 'odd' behaviour in images.
The idea is that if we detect odd behaviour for an extended period (10-20 sec?)
that this may be indicative of a partial seizure.

We are using a neural network to determine whether an image is 'normal' or 'odd'

But first we have to train the neural network with example images.....

Proportion of negative (=normal) samples compared to positive (odd) ones
should be 16 to 18% (ie a lot less normal ones than odd):  https://pdfs.semanticscholar.org/235d/cf221298ae74252b10ef3f69acd4f7e1585f.pdf

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
