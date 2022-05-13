# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:30:42 2021

@author: Atesh
"""
import sys
import cv2
import os
from sys import platform
import argparse
import time

import pytorch_openpose.lib.openpose37.pyopenpose as op
#import pytorch_openpose.lib.openpose.pyopenpose as op


class Hand:
    def __init__(self,model_folder):
        try:
            # Import Openpose (Windows/Ubuntu/OSX)
#            dir_path = os.path.dirname(os.path.realpath(__file__))
            

    
    #        # Flags
    #        parser = argparse.ArgumentParser()
    #        parser.add_argument("--image_path", default="../examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    #        args = parser.parse_known_args()
        
            # Custom Params (refer to include/openpose/flags.hpp for more parameters)
            self.params = dict()
            self.params["model_folder"] = model_folder
            self.params["hand"] = True
#            self.params["hand_detector"] = 2
#            self.params["body"] = 1
            
            


        
    #        # Add others in path?
    #        for i in range(0, len(args[1])):
    #            curr_item = args[1][i]
    #            if i != len(args[1])-1: next_item = args[1][i+1]
    #            else: next_item = "1"
    #            if "--" in curr_item and "--" in next_item:
    #                key = curr_item.replace('-','')
    #                if key not in params:  params[key] = "1"
    #            elif "--" in curr_item and "--" not in next_item:
    #                key = curr_item.replace('-','')
    #                if key not in params: params[key] = next_item
        
            # Construct it from system arguments
            # op.init_argv(args[1])
            # oppython = op.OpenposePython()


#        self.handRectangles = [
#        # Left/Right hands person 0
#        [
#        op.Rectangle(320.035889, 377.675049, 69.300949, 69.300949),
#        op.Rectangle(0., 0., 0., 0.),
#        ],
#        # Left/Right hands person 1
#        [
#        op.Rectangle(80.155792, 407.673492, 80.812706, 80.812706),
#        op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
#        ],
#        # Left/Right hands person 2
#        [
#        op.Rectangle(185.692673, 303.112244, 157.587555, 157.587555),
#        op.Rectangle(88.984360, 268.866547, 117.818230, 117.818230),
#        ]
#    ]        
#        self.handRectangles = [[op.Rectangle(320.035889, 377.675049, 69.300949, 69.300949), op.Rectangle(0., 0., 0., 0.),]]
    
        
            # Starting OpenPose
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(self.params)
            self.opWrapper.start()
            # Create new datum
            self.datum = op.Datum()
            
            
#            self.body_opWrapper = op.WrapperPython()
#            self.opWrapper.configure(self.params)
#            self.opWrapper.start()
#            # Create new datum
#            self.datum = op.Datum()
            
        except Exception as e:
            print(e)
            sys.exit(-1)
    def __call__(self,oriImg):
        # Read image and face rectangle locations
#        imageToProcess = cv2.imread(args[0].image_path)


        self.datum.cvInputData = oriImg

        

        
        self.opWrapper.emplaceAndPop([self.datum])
#        self.datum.poseKeypoints[0][4,0:2]
#        self.datum.poseKeypoints[0][7,0:2]
        
#        self.datum.handRectangles = self.handRectangles
#        self.handRectangles = [
#        # Left/Right hands person 0
#        [
#        op.Rectangle(self.datum.poseKeypoints[0][4,0], self.datum.poseKeypoints[0][4,0], 69.300949, 69.300949),
#        op.Rectangle(self.datum.poseKeypoints[0][7,0], self.datum.poseKeypoints[0][7,0], 0., 0.),
#        ]
#    ]
#        import pdb
#        pdb.set_trace()
#        # Process and display image
#        self.opWrapper.emplaceAndPop([self.datum])
#        print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
#        print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
#        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
#        cv2.waitKey(0)
        return(self.datum.handKeypoints)
