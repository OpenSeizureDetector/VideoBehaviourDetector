#!/usr/bin/python

import cv2
from getImage import getImage
#from cv2 import cv
import urllib
import httplib2                     # Needed to communicate with camera
import numpy as np
import time
import datetime
import os

outDir = "./images"


if (os.path.exists(outDir)):
    if (os.path.isdir(outDir)):
        print "Output Directory %s already exists - OK" % outDir
    else:
        print "ERROR:  %s exists, but is not a Directory" % outDir
        exit(-1)
else:
    print "Creating output directory %s" % outDir
    os.makedirs(outDir)



while True:
    frame = getImage()
    if frame is None:
        print 'Error Retrieving Image'
        #exit(-1)
    else:
        tVal = datetime.datetime.now()
        fname = "img_%s.png" % (tVal.strftime("%Y%m%d%H%M%S"))
        print "fname=%s" % (fname)
        cv2.imwrite(os.path.join(outDir,fname),frame)
        
    time.sleep(10)
