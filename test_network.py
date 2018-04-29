#!/usr/bin/python

# USAGE
# python test_network.py --model <filename> --dataset <folder path>
# Based on https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils import paths
import cv2
import random
from getImage import getImage
import time
import os, sys

# Size of image to use (input images are scaled to this size)
# Note that I THINK this has to be the same as was used to train the model.
IM_SIZE = (100,100)


def classifyImage(model,image,gui=False):
        """ Classify the image using the image using the given model.
        if gui is True, displays it on the screen.
        """
        orig = image.copy()

        # pre-process the image for classification
        image = cv2.resize(image, IM_SIZE)
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify the input image
        (normal, odd) = model.predict(image)[0]

        # build the label
        label = "Odd" if odd > normal else "Normal"
        label = "{}: P(odd)={:.1f}%".format(label, odd * 100)

        # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

        # show the output image
        if (gui): cv2.imshow("Output", output)

        return(odd,output)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-d", "--dataset", required=False,
	help="path to input image")
ap.add_argument("-l", "--live", required=False, action="store_true",
        help="Use live images from the camera rather than files")
ap.add_argument("-g", "--gui", required=False, action="store_true",
        help="Show a simple graphical interface that displays the images and analysis results")
args = vars(ap.parse_args())

print args

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

if not args['live']:
        # grab the image paths and randomly shuffle them
        imagePaths = sorted(list(paths.list_images(args["dataset"])))
        random.seed(42)
        random.shuffle(imagePaths)

        nNorm = 0
        nNormErr = 0
        nOdd = 0
        nOddErr = 0
        
        # loop over the input images
        for imagePath in imagePaths:
                # load the image, 
                image = cv2.imread(imagePath)
                odd,outimg = classifyImage(model,image,args['gui'])
                if (args['gui']):
                        cv2.waitKey(1000)
                        if (odd>0.5):
                                label = "**** ODD *****"
                        else:
                                label = "Normal"
                        label = "{}: P(odd)={:.0f}%".format(label, odd * 100)
                        print("imagePath=%s" % imagePath)
                        print (label)
                else:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                # We expect all images to be in a directory named its
                # class name (e.g. 'normal' or 'odd'
                imgClass = imagePath.split(os.path.sep)[-2]
                if (imgClass == 'odd'):
                        nOdd+=1
                        if (odd<0.5):  nOddErr+=1
                if (imgClass == 'normal'):
                        nNorm+=1
                        if (odd>=0.5): nNormErr+=1

        print ("")
        print ("************************")
        print ("*     RESULTS          *")
        print ("************************")
        print ("")
        print ("Number of Odd Images Tested = %d" % nOdd)
        print ("Number of Odd Image Errors  = %d" % nOddErr)
        if (nOdd>0):
                print ("  Odd Detection Reliability = %d%%" % (100-int(100*nOddErr/nOdd))) 
        print ("")
        print ("Number of Normal Images Tested = %d" % nNorm)
        print ("Number of Normal Image Errors  = %d" % nNormErr)
        if (nNorm>0):
                print ("  Normal Detection Reliability = %d%%" % (100-int(100*nNormErr/nNorm)))
        print ("")

else:
        while True:
            imGray = getImage()
            #image = np.zeros((IM_SIZE[0],IM_SIZE[1],3), np.uint8)
            image = cv2.cvtColor(imGray,cv2.COLOR_GRAY2BGR)
            if image is None:
                print 'Error Retrieving Image'
            else:
                odd,outimg = classifyImage(model,image,args['gui'])
                if (args['gui']):
                        k= cv2.waitKey(1)
                        if k==0x1b:
                            print 'Esc. Exiting'
                            break
                if (odd>0.5):
                        label = "**** ODD *****"
                else:
                        label = "Normal"
                label = "{}: P(odd)={:.0f}%".format(label, odd * 100)
                print (label)
            time.sleep(1)

