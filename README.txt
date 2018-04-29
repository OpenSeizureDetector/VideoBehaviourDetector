README
======

This is an experimental attempt at detection of 'odd' behaviour in images.
The idea is that if we detect odd behaviour for an extended period (10-20 sec?)
that this may be indicative of a partial seizure.

We are using a neural network to determine whether an image is 'normal' or 'odd'

But first we have to train the neural network with example images.....

Proportion of negative (=normal) samples compared to positive (odd) ones
should be 16 to 18% (ie a lot less normal ones than odd):  https://pdfs.semanticscholar.org/235d/cf221298ae74252b10ef3f69acd4f7e1585f.pdf
