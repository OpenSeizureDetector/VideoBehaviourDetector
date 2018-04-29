#!/usr/bin/python

# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png
# Based on https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import time
from getImage import getImage

# Size of image to use (input images are scaled to this size)
IM_SIZE = (100,100)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])


while True:
    imGray = getImage()
    #image = np.zeros((IM_SIZE[0],IM_SIZE[1],3), np.uint8)
    image = cv2.cvtColor(imGray,cv2.COLOR_GRAY2BGR)
    if image is None:
        print 'Error Retrieving Image'
    else:
        #tVal = datetime.datetime.now()
        #fname = "img_%s.png" % (tVal.strftime("%Y%d%m%H%M%S"))
        #print "fname=%s" % (fname)
        #cv2.imwrite(os.path.join(outDir,fname),frame)

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
        proba = odd if odd > normal else normal
        label = "{}: {:.2f}%".format(label, proba * 100)

        # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

        # show the output image
        cv2.imshow("Output", output)
        k= cv2.waitKey(1)
        if k==0x1b:
            print 'Esc. Exiting'
            break

    time.sleep(1)

