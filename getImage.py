#!/usr/bin/python




import cv2
#from cv2 import cv
import urllib
import httplib2                     # Needed to communicate with camera
import numpy as np
import time
import sys

def getImage():
    """ Use HTTP GET request to retrieve a jpeg image from the camera.
    Returns an OpenCV greyscale image, or None if there was an error.
    """
    #print "getOpenSeizureDetectorData"
    h = httplib2.Http(".cache")
    h.add_credentials('guest', 'guest')
    requestStr = "http://192.168.0.6/image/jpeg.cgi"
    #print requestStr
    try:
        resp, content = h.request(requestStr,
                                  "GET")
        if resp['status'] == '200':
            buf = np.fromstring(content,dtype=np.uint8)
            img = cv2.imdecode(buf,0)    
            return img
        else:
            print resp,content,resp
            return None
    except:
        print "getImage Error:",sys.exc_info()[0]
        return None


if __name__ == "__main__":
    cv2.namedWindow("Frame",1)
    cv2.namedWindow("Mask",1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        frame = getImage()
        if frame is None:
            print 'Cam not found'
            exit(-1)
        else:
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            #print "Displaying image..."
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", fgmask)
            k= cv2.waitKey(1)
        if k==0x1b:
            print 'Esc. Exiting'
            break
        time.sleep(1)
