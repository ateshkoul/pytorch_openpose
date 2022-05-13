# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import pdb
import time
#import pytorch_openpose.lib.openpose.pyopenpose as op
import pytorch_openpose.lib.openpose37.pyopenpose as op
class Body:
    def __init__(self,model_folder):
        try:
            # Import Openpose (Windows/Ubuntu/OSX)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            try:
                # Windows Import
                if platform == "win32":
                    # Change these variables to point to the correct folder (Release/x64 etc.)
                    # sys.path.append(dir_path + '/../../python/openpose/Release');
                    # sys.path.append('Y:\\Understanding intentions\\neural-networks\\Global_resources\\pytorch_openpose\\lib\\openpose');
                    # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                    # os.environ['PATH']  = os.environ['PATH'] + ';'  +  'Y:\\Understanding intentions\\neural-networks\\Global_resources\\pytorch_openpose\\lib\\openpose\\bin;'

                    # pdb.set_trace()
#                    import pytorch_openpose.lib.openpose.pyopenpose as op
                    import pytorch_openpose.lib.openpose37.pyopenpose as op

                    # import pyopenpose as op
                else:
                    # Change these variables to point to the correct folder (Release/x64 etc.)
                    sys.path.append('../../python');
                    # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                    # sys.path.append('/usr/local/python')
                    from openpose import pyopenpose as op
            except ImportError as e:
                print(
                    'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
                raise e

            # Flags
            # parser = argparse.ArgumentParser()
            # parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg",
            #                     help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
            # args = parser.parse_known_args()

            # Custom Params (refer to include/openpose/flags.hpp for more parameters)
            self.params = dict()
            self.params["model_folder"] = model_folder

            # Construct it from system arguments
            # op.init_argv(args[1])
            # oppython = op.OpenposePython()
            # pdb.set_trace()
            # Starting OpenPose
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(self.params)
            self.opWrapper.start()

            # Process Image
            self.datum = op.Datum()
        except Exception as e:
            print(e)
            sys.exit(-1)


        # try:
        #     # Import Openpose (Windows/Ubuntu/OSX)
        #     dir_path = os.path.dirname(os.path.realpath(__file__))
        #     try:
        #         # Windows Import
        #         if platform == "win32":
        #             # Change these variables to point to the correct folder (Release/x64 etc.)
        #             # sys.path.append(dir_path + '/../../python/openpose/Release');
        #             # sys.path.append('Y:\\Understanding intentions\\neural-networks\\Global_resources\\pytorch_openpose\\lib\\openpose');
        #             # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        #             # os.environ['PATH']  = os.environ['PATH'] + ';'  +  'Y:\\Understanding intentions\\neural-networks\\Global_resources\\pytorch_openpose\\lib\\openpose\\bin;'
        #
        #             # pdb.set_trace()
        #             import pytorch_openpose.lib.openpose.pyopenpose as op
        #             # import pyopenpose as op
        #         else:
        #             # Change these variables to point to the correct folder (Release/x64 etc.)
        #             sys.path.append('../../python');
        #             # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        #             # sys.path.append('/usr/local/python')
        #             from openpose import pyopenpose as op
        #     except ImportError as e:
        #         print(
        #             'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        #         raise e
        #
        # # Flags
		# #parser = argparse.ArgumentParser()
		# #parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
		# #args = parser.parse_known_args()
		# #pdb.set_trace()
		# # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        # self.params = dict()
        # #params["model_folder"] = "../../models/"
        # self.params["model_folder"] = model_folder
        # self.opWrapper = op.WrapperPython()
        # self.opWrapper.configure(self.params)
        # self.opWrapper.start()
        #
        # # Process Image
        # self.datum = op.Datum()
        #     # Add others in path?
        #     #for i in range(0, len(args[1])):
        #     #    curr_item = args[1][i]
        #     #    if i != len(args[1])-1: next_item = args[1][i+1]
        #     #    else: next_item = "1"
        #     #    if "--" in curr_item and "--" in next_item:
        #     #        key = curr_item.replace('-','')
        #     #        if key not in params:  params[key] = "1"
        #     #    elif "--" in curr_item and "--" not in next_item:
        #     #        key = curr_item.replace('-','')
        #     #        if key not in params: params[key] = next_item
        #
        #     # Construct it from system arguments
        #     # op.init_argv(args[1])
        #     # oppython = op.OpenposePython()
        #
        #     # Starting OpenPose
        #
        # except Exception as e:
        #     print(e)
        #     sys.exit(-1)
    def __call__(self,oriImg):
        #sys.path.append('D:\\Software\\Neural_net\\openpose-master\\python\\openpose\\Release');					
        #import pytorch_openpose.lib.openpose.pyopenpose as op
        #pdb.set_trace()
        # to be consistent with the previous functions, I input the data from cv2.imread
        #imageToProcess = cv2.imread(image_path)  
        #pdb.set_trace()
        # In case, there is a transformation, this has to be corrected, pytorch uses 
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # not used here because I use a different transform that doesn't change axis
        # Also, very important, the data has to be as uint8 (not float, not long)
        #self.datum.cvInputData = oriImg.transpose((2,0, 1))	
        self.datum.cvInputData = oriImg
        #start = time.time()
        self.opWrapper.emplaceAndPop([self.datum])
        #pdb.set_trace()
        #print("time is ",time.time()-start)
        # Display Image
        #print("Body keypoints: \n" + str(self.datum.poseKeypoints))
        
        #pdb.set_trace()
        #cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", self.datum.cvOutputData)
        #cv2.waitKey(0)
        return(self.datum.poseKeypoints)
