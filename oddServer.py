#!/usr/bin/python

import bottle
import json
import os
from datetime import datetime
import threading

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

bottle.debug(True)

wwwPath = "."

# Size of image to use (input images are scaled to this size)
# Note that I THINK this has to be the same as was used to train the model.
IM_SIZE = (100,100)

ODD_LIST_LEN = 5

##################################################################
# Image Processing Thread
##################################################################
class processImagesThread (threading.Thread):
    def __init__(self, modelFname):
        threading.Thread.__init__(self)
        self.modelFname = modelFname

    def classifyImage(self,image,gui=False):
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
        (normal, odd) = self.model.predict(image)[0]

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


    def run(self):
        self.oddCount = 0
        self.oddList = []
        # load the trained convolutional neural network
        print("[INFO] loading network...")
        self.model = load_model(self.modelFname)
        print "Starting thread"
        while(1):
            imGray = getImage()
            image = cv2.cvtColor(imGray,cv2.COLOR_GRAY2BGR)
            if image is None:
                print 'Error Retrieving Image'
            else:
                odd,self.outImg = self.classifyImage(image,True)
                self.lastTime = datetime.now()
                self.oddList.append(odd)
                if len(self.oddList)>ODD_LIST_LEN:
                    self.oddList = self.oddList[1:]
                self.meanOdd = sum(self.oddList)/len(self.oddList)
                print self.oddList,self.meanOdd
            time.sleep(1)
        print "Exiting Thread"

    def getData(self):
        data = {'meanOdd':self.meanOdd,'time':self.lastTime.strftime("%Y%m%d%H%M%S")}
        return data
    def getImageData(self):
        return self.outImg

def jsonp(request,d):
    if (request.query.callback):
        return "%s(%s)" % (request.query.callback,d)
    return d

##################################################################
# GET requests 
##################################################################
@bottle.get('/data')
#@bottle.get('/data/<idStr>')
def data(idStr=None):
    print "data.get - idStr=%s." % idStr
    bottle.response.content_type='application/javascript'
    return jsonp(bottle.request,imgThread.getData())
@bottle.get('/image')
def image():
    print "image.get"
    imgbin =  imgThread.getImageData()
    response = bottle.BaseResponse()
    response.content_type = 'image/png'
    response.status=200
    response.body=imgbin.tobytes()
    return response


####################################################
# PUT requests 
####################################################
#@bottle.put('/risks/<idStr>')
#def risks(idStr):
#    print "risks.put"
#    print bottle.request
#    print bottle.request.json



#######################################################
# POST request 
#######################################################
#@bottle.post('/risks')
#def risks():
#    print "risks.post"
#    print bottle.request
#    if 'data' in bottle.request.json:
#    return(jsonp(bottle.request,getRiskJson(str(rowId))))


##################################################################
# DELETE requests 
##################################################################
#@bottle.delete('/risks/<idStr>')
#def risks(idStr):
#    print "risks.delete %s" % idStr

#@bottle.get('/login')
#@bottle.post('/login')
#def login():
    # Fixme - authenticate user and return a token that is valid for a
    # specific session.
#    keyStr = "authentication_key_string";
#    return(jsonp(bottle.request,"{msg:'ok',key:'"+keyStr+"'}"))
    
#@bottle.get('/logout')
#@bottle.post('/logout')
#def logout():
#    # Fixme - logout the user and delete the token.
#    return(jsonp(bottle.request,"{msg:'ok'}"))
    

#################################################################
# Entry Page
#################################################################

@bottle.route('/')
#@bottle.route('/index.html')
def entry_page():
    #bottle.redirect("/static/index.html")
    #bottle.redirect("/app/index.html")
    bottle.redirect("/index.html")

#################################################################
# Static files (=main web app)
#################################################################
@bottle.route('/<filename:path>')
def static_files(filename="risk.html"):
    print "static_files - filename = %s." % (filename) 
    return bottle.static_file(filename,root=wwwPath)




if (__name__=="__main__"):
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
            help="path to trained model model")
    args = vars(ap.parse_args())

    print args




    #####################################
    # Start the image processing thread
    #####################################
    imgThread = processImagesThread(args["model"])
    imgThread.start()
    
    ########################################################
    # Run the server
    ########################################################
    bottle.run(host='localhost', port=8080)
