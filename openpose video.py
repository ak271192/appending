# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:25:30 2019

@author: dpotti
"""

# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
from PIL import Image
import math
from scipy.spatial.distance import pdist
from operator import sub
import imutils as im
#maskPath = "C:/Users/Dpotti/Desktop/pant5.png"
#mask = Image.open(maskPath).convert("RGBA")
##gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY )
##background = Image.fromarray(mask)
#maskPath = "C:/Users/Dpotti/Desktop/pant5.png"
#
##face_cascade = cv2.CascadeClassifier('C:/Users/Dpotti/haarcascade_fullbody.xml')
##low_cascade = cv2.CascadeClassifier('C:/Users/Dpotti/haarcascade_lowerbody.xml')
##upp_cascade = cv2.CascadeClassifier('C:/Users/Dpotti/haarcascade_upperbody.xml')
#mask = Image.open(maskPath).convert("RGBA")
#cap = cv.VideoCapture(0)
#def dress(image):
#
#    
#    
#    ret, frame = cap.read()
#    background = Image.fromarray(image)

    #gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY )

    #faces = face_cascade.detectMultiScale(gray, 1.1 , 4)
    #low = low_cascade.detectMultiScale(gray, 1.1 , 3)
    #upp = upp_cascade.detectMultiScale(gray, 1.1 , 4)
#    for (x,y,w,h) in POSE_PAIRS:
#        resized_mask = mask.resize((w,h), Image.ANTIALIAS)
#        offset = (x,y)
#        background.paste(resized_mask, offset, mask=resized_mask)
#        return np.asarray(background)
    

    

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=268, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=268, type=int, help='Resize input to specific height.')

args = parser.parse_args()
#Pre-trained models for Human Pose Estimation

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 } #, "LRhand":19, "LLhand": 20}

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["LShoulder", "LHip"], ["RShoulder", "RHip"],
               ["RHip", "LHip"], #["Neck", "RHip"],  #["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

#final_list=list(BODY_PARTS.values())
#print(final_list)



#dist = numpy.linalg.norm("RShoulder"-"RElbow")

maskPath = "C:/Users/Dpotti/Desktop/Datasets/fullhand.png"
maskPath1 = "C:/Users/Dpotti/Desktop/pant5.png"
#image = cv.imread("C:/Users/Dpotti/Downloads/human-pose-estimation-opencv-master/human-pose-estimation-opencv-master/image.jpg")
#(h, w, d) = image.shape
##image = Image.open(maskPath).convert("RGBA")
#
#image = im.resize(image, width=100, height=100)
#
#
## Gray scale conversion for processing
##gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
inWidth = args.width
inHeight = args.height
mask = Image.open(maskPath).convert("RGBA")
mask1 = Image.open(maskPath1).convert("RGBA")
#background = Image.open(args.input).convert("RGBA")
weight,height = mask.size
#w,h = (300,300)
#x,y = (200,100)

#d = RShoulder - RElbow
#d1 = RShoulder - LShoulder
#s1 =  inHeight/ inWidth
#
#w1 = inWidth* d2/d1
#h1 = w1 * s1
#Use the getModels.sh file provided with the code to download all 
#the model weights to the respective folders. 
#Step 2: Load Network
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture(args.input if args.input else 0)


while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
   # Read Image and Prepare Input to the Network
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
  # Make Predictions and Parse Keypoints
 # Once the image is passed to the model, the predictions can be made using a single line of code. The forward method for the DNN class in OpenCV makes a forward pass through the network which is 
 # just another way of saying it is making a prediction.
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]
        print(heatMap)

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        # Find global maxima of the probMap.
        _, conf, _, point = cv.minMaxLoc(heatMap)
     #   Scale the point to fit on the original image
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
       # dist = np.linalg.norm(x-y)
      #  int(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)
        print(points)
        #print(points[2])
#        list(points[0])
#        list(points[14])
#       # plot = list(points)
#       # plot2 =  list(points[14])
#      
##        print(points)
#        print(points[0])
##        print(points[1])
#        print(points[2])
#        print(points[14])
#        print(points[15])
       # eucli = math.sqrt( (plot1[0]-plot2[0])**2)# + (plot1[1]-plot2[1])**2 )
       # dis = sqrt((points[0] - points[0]) **2 + (points[14] - points[14]) **2)
#        pairwise_distances = pdist(BODY_PARTS, metric="euclidean", p=2)
#        print(pairwise_distances)
    #    sum_dims = sum((x - y) ** 2 for x, y in zip(int(x), int(y)))
#        background = Image.open("hman.jpg")
#        foreground = Image.open("shirt1.png")
#        Image.alpha_composite(background, foreground).save("test3.png")
#        background.paste(foreground, (0, 0), foreground)
#        background.show()

    for pair in POSE_PAIRS:
        #look for two connected body parts
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            final = cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
#            pairwise_distances = pdist(points, metric="euclidean", p=2)
#            print(pairwise_distances)
            
            #distance = math.sqrt((points[idFrom]-points[idTo])**2)

           # cv::sqrt(diff.points[idFrom]*diff.points[idFrom] + diff.points[idTo]*diff.points[idTo])
      #      dis = sqrt(pow(points[idFrom] - points[idTo]))
            final1 = cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
#            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    


    t, _ = net.getPerfProfile()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    
#    for x in points:
#        #print(x,"in x")
#        print(x,"bfre none")
#        if x == None:
#            print(x,"before loop")
#            x = (0,0)
#            print(x,"after loop")
#            print(points)
            
            
            #print(points[x])
        #points[5] = (0,0)
    if points[2] is None:
        w = (0,0)
#            rshoulder = points[2]
#            print("rshoulderif", rshoulder)
#            lshoulder = points[5]
#            print("lshoulderif", lshoulder)
#            w = tuple(map(sub, lshoulder, rshoulder))        
        print("Please stand properly to detect the width of shoulder")
    elif points[5] is None:
        w = (0,0)
        print("Please stand properly to detect the width of hip")
            
    else:
        rshoulder = points[2]
        print("rshoulder", rshoulder)
        lshoulder = points[5]
        print("lshoulder", lshoulder)
        w = tuple(map(sub, lshoulder, rshoulder))
        print(w)
        print("detected width") 
            
    if points[8] is None or w == (0,0):
        h = (0,0)
#        c = w
#        d = (30,30)
#        h = tuple(map(sum, c, d))
#        w,h = tuple(map(sub, w, h))
        #rhip =  w + 30
#        h = tuple(map(sub, rshoulder, rhip))
#        print(h)
#        w,h = tuple(map(sub, w, h))
        print("Please stand properly to detect the height")
    else:
        rhip = points[8]
        print("rshoulder", rshoulder)
        
        print("lshoulder", rhip)
        h = tuple(map(sub, rshoulder, rhip))
        print(h)
        print("detected height")
        
        w = int(w*1.7)
        h = int(h*1.3)
        w,h = tuple(map(sub, w, h))
    
        
    
    
    
#    for (x,y,w,h) in pair:
#        resized_mask = mask.resize((w,h), Image.ANTIALIAS)
#        offset = (x,y)
#        background.paste(resized_mask, offset, mask=resized_mask)
    #cv.imshow('show',frame)
        def new(image):
        
            background = Image.fromarray(frame)
            
            if (w,h) <= (0,0):
                print("Please stand properly fast")
            else:
                resized_mask = mask.resize((w,h), Image.ANTIALIAS)
#                a = points[2]
#                b = (63,63)
#                offset = tuple(map(sub, a, b))
#                print(offset)
                offset = points[2]
       #     offset = points[2]
                background.paste(resized_mask, offset, mask=resized_mask)
#        resized_mask1 = mask1.resize((360,440), Image.ANTIALIAS)
#        offset1 = points[8]
#   # mask = Image.open(maskPath).convert("RGBA")
#        background.paste(resized_mask1, offset1, mask=resized_mask1)
#    background = np.asarray(background)
#   # background.paste(foreground, (99, 105), foreground)
#    cv.imshow('Live', background)
        
                return np.asarray(background)
    
        cv.imshow('show',frame)
        cv.imshow('show1',new(frame))

    
    
   # def dress(image):
#
#    
#        ret, frame = cap.read()
#
#        #gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY )
#
#    #faces = face_cascade.detectMultiScale(gray, 1.1 , 4)
#        #low = low_cascade.detectMultiScale(gray, 1.1 , 3)
#    #upp = upp_cascade.detectMultiScale(gray, 1.1 , 4)
      #  background = Image.fromarray(image)
#
  #     for (x,y,w,h) in points:
#       w = 1279
#       h = 849
#       resized_mask = mask.resize((w,h), Image.ANTIALIAS)
  #############################################################
#    foreground = Image.open("C:/Users/Dpotti/Desktop/shirt1.png")
#    #width,height
#    resized_mask = mask.resize((300,300), Image.ANTIALIAS)
#    a = points[2]
#    b = (50,50)
#    offset = tuple(map(sub, a, b))
#    print(offset)
#    #offset = points[2]
#   # mask = Image.open(maskPath).convert("RGBA")
#    background.paste(resized_mask, offset, mask=resized_mask)
#    background.show()
#    background.paste(foreground, (99, 105), foreground)
#    background = np.asarray(background)
#    cv.imshow('Live', background)
  ############################################################
#    foreground = Image.open("C:/Users/Dpotti/Desktop/pant5.png")
#    resized_mask1 = mask1.resize((360,440), Image.ANTIALIAS)
##    a = points[8]
##    b = (63,63)
##    offset1 = tuple(map(sub, a, b))
##    print(offset1)
#    offset1 = points[8]
#   # mask = Image.open(maskPath).convert("RGBA")
#    background.paste(resized_mask1, offset1, mask=resized_mask1)
##    background = np.asarray(background)
##   # background.paste(foreground, (99, 105), foreground)
##    cv.imshow('Live', background)
#    background.show()
#  
#  
#    foreground = Image.open("C:/Users/Dpotti/Desktop/shirt1.png")
#    #width,height
#    resized_mask = mask.resize((300,300), Image.ANTIALIAS)
#    offset = (258,120)
#   # mask = Image.open(maskPath).convert("RGBA")
#    background.paste(resized_mask, offset, mask=resized_mask)
#    background.show()
#    final = np.asarray(background)
#   # background.paste(foreground, (99, 105), foreground)
##    background = np.asarray(background)
#    cv.imshow('Live', final)
##  
##  ############################shirt#######################
#    foreground = Image.open("C:/Users/Dpotti/Desktop/shirt1.png")
#    #width,height
#    resized_mask = mask.resize((300,300), Image.ANTIALIAS)
#    #right,down
#    offset = points[2]
#   # mask = Image.open(maskPath).convert("RGBA")
#    background.paste(resized_mask, offset, mask=resized_mask)
#   # background.paste(foreground, (99, 105), foreground)
##    background = np.asarray(background)
##    cv.imshow('Live', background)
#    background.show()
#    
##    #############################pant############################
#    foreground = Image.open("C:/Users/Dpotti/Desktop/pant5.png")
#    resized_mask1 = mask1.resize((360,440), Image.ANTIALIAS)
#    offset1 = points[8]
#   # mask = Image.open(maskPath).convert("RGBA")
#    background.paste(resized_mask1, offset1, mask=resized_mask1)
##    background = np.asarray(background)
##   # background.paste(foreground, (99, 105), foreground)
##    cv.imshow('Live', background)
#    background.show()
#  ##################################################
#  #  np.asarray(background)
#      
##            offset = (int(x),int(y))
##            background.append(resized_mask, offset, mask=resized_mask)
##        return np.asarray(background)
##    while True:
##	# read return value and frame
##    	ret, frame = cap.read()
##
#    	if ret == True:
#            cv.imshow('Live', dress(frame))
#            if cv.waitKey(1) == 27:
#                breakana
## release cam
#cap.release()
## destroy all open opencv windows
#cv.destroyAllWindows()

#     points(sqrt(["RShoulder"]^2 + ["RElbow"]^2))
#    #cv.imshow('OpenPose using OpenCV', dress(frame))
#  
   # distance = out(sqrt({BODY_PARTS["RShoulder"]**2 + BODY_PARTS["RElbow"]**2}))
#    
    
    