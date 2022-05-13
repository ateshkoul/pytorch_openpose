import matplotlib.pyplot as plt
#from pytorch_openpose.poses.body import Body
from pytorch_openpose.poses.body_api import Body
#from pytorch_openpose.poses.hand import Hand
from pytorch_openpose.poses.hand_api import Hand

#from pytorch_openpose.poses.face import Face
from pytorch_openpose.poses.face_api import Face
import cv2
import pdb
import numpy as np
import pdb
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

class pose_label():
    def __init__(self,pred_type,model_folder='..\\..\\Global_resources\\pytorch_openpose\\models\\',saveFig = False,figName='openpose_image.tif'):
        self.pred_type = pred_type
        self.figName = figName
        self.saveFig = saveFig
        for case in switch(self.pred_type):
            if case('body'):
                # for api
                #self.body_model = Body(model_folder)                
#                pdb.set_trace()
#                self.body_model = Body(model_folder+'body_pose_model.pth')
                self.body_model = Body(model_folder)

                break
            if case('hand'):
#                self.hand_model = Hand(model_folder+'hand_pose_model.pth')
                self.hand_model = Hand(model_folder)

                break
            if case('face'):
#                pdb.set_trace()
#                self.face_model = Face(model_folder+'face_pose_model.pt')
                # for api
                self.face_model = Face(model_folder)
                break
            if case('all'):
                self.body_model = Body(model_folder+'body_pose_model.pth')
                self.face_model = Face(model_folder+'face_pose_model.pt')
                self.hand_model = Hand(model_folder+'hand_pose_model.pth')
                break
            if case('body_face'):
#                self.body_model = Body(model_folder+'body_pose_model.pth')
#                self.face_model = Face(model_folder+'face_pose_model.pt')
                self.body_model = Body(model_folder)
                self.face_model = Face(model_folder)
#                self.hand_model = Hand(model_folder+'hand_pose_model.pth')
                break
            if case():
                print("wrong model chosen")
#                pdb.set_trace()
                break       
    
    

        
    def predict_frame(self,oriImg):
                # done for webcam
        oriImg_dat = oriImg 
        #oriImg_dat = cv2.imread(oriImg)
        #plt.imshow(oriImg[:,:,[2,1,0]])
        # only for the bottle images
        plt.imshow(oriImg_dat/255)
        #pdb.set_trace()
        for case in switch(self.pred_type):
            if case('body'):
                peaks_body = self.body_model(oriImg_dat)
                #peaks_body,subset = self.body_model(oriImg)
                plt.scatter(peaks_body[0][:,0],peaks_body[0][:,1])
                
#                pdb.set_trace()
                for x in range(0,peaks_body[0].shape[0]):
                    plt.text(peaks_body[0][x,0],peaks_body[0][x,1],str(peaks_body[0][x,3]))                
                break
            if case('hand'):
                peaks_hand = self.hand_model(oriImg_dat)
                plt.scatter(peaks_hand[:,0],peaks_hand[:,1])
                break
            if case('face'):
#                pdb.set_trace()
                peaks_face = self.face_model(oriImg_dat)
                plt.scatter(peaks_face[:,0],peaks_face[:,1])
                break
            if case('all'):
                peaks_body,subset = self.body_model(oriImg)
                peaks_hand = self.hand_model(oriImg)
                peaks_face = self.face_model(oriImg)
                plt.scatter(peaks_body[:,0],peaks_body[:,1])
                plt.scatter(peaks_hand[:,0],peaks_hand[:,1])
                plt.scatter(peaks_face[:,0],peaks_face[:,1])
                break
            if case():
                print("wrong model chosen")
                break 
        x,y,c = oriImg_dat.shape
        plt.axis([0, y, x,0])
        
        if (self.saveFig ==True):
            plt.savefig(self.figName)
        plt.show()
        
    def return_predictions(self,oriImg):
        # done for webcam
        oriImg_dat = oriImg 
        #oriImg_dat = cv2.imread(oriImg)
        #plt.imshow(oriImg[:,:,[2,1,0]])
        # only for the bottle images
#        plt.imshow(oriImg_dat/255)
        #pdb.set_trace()
        for case in switch(self.pred_type):
            if case('body'):
#                pdb.set_trace()
                peaks_body = self.body_model(oriImg_dat)
                
#                pdb.set_trace()
                formated_peaks_x = np.repeat(np.nan,26)
                formated_peaks_y = np.repeat(np.nan,26)
                
#                try:
#                    len(peaks_body[0])
#                except:
#                        pdb.set_trace()
                # check if there are peaks to work with:
#                pdb.set_trace()
                if len(peaks_body.shape)>0:
                    # a very crude way to deal with this right now. 
                    #  >26 are because it seems the algorithm has found multiple people
                    if peaks_body[0].shape[0]>26:
                        formated_peaks_x = peaks_body[0][0:26,0]                    
                        formated_peaks_y = peaks_body[0][0:26,1]
                    else:
#                        pdb.set_trace()
                        formated_peaks_x = peaks_body[0][:,0]                    
                        formated_peaks_y = peaks_body[0][:,1]
                # not for api
#                # check if there are peaks to work with:
#                if len(peaks_body[0])>0:
#                    # a very crude way to deal with this right now. 
#                    #  >26 are because it seems the algorithm has found multiple people
#                    if peaks_body[0].shape[0]>26:
#                        formated_peaks_x[np.array(peaks_body[0][0:26,3],dtype=int)] = peaks_body[0][0:26,0]                    
#                        formated_peaks_y[np.array(peaks_body[0][0:26,3],dtype=int)] = peaks_body[0][0:26,1]
#                    else:
##                        pdb.set_trace()
#                        formated_peaks_x[np.array(peaks_body[0][:,3],dtype=int)] = peaks_body[0][:,0]                    
#                        formated_peaks_y[np.array(peaks_body[0][:,3],dtype=int)] = peaks_body[0][:,1]
                formated_peaks = np.hstack([formated_peaks_x,formated_peaks_y])
                return(formated_peaks)

            if case('hand'):
#               To be implemented as a long list?
                peaks_hand = self.hand_model(oriImg_dat)
#                pdb.set_trace()
                # 21*2 for 2 hands
                formated_peaks_x = np.repeat(np.nan,21*2)
                formated_peaks_y = np.repeat(np.nan,21*2)
                # for api:
                # only if hand is detected:
                if len(peaks_hand[0].shape)>0:
                    try:
                        peaks_hand_left = peaks_hand[0][0]             
                        peaks_hand_right = peaks_hand[1][0]  
        #                formated_peaks_x = np.repeat(np.nan,70) 
                        formated_peaks_x = np.hstack([peaks_hand_left[:,0],peaks_hand_right[:,0]])
                        
        #                formated_peaks_y = np.repeat(np.nan,70) 
                        formated_peaks_y = np.hstack([peaks_hand_left[:,1],peaks_hand_right[:,1]])
                    except:
                        pdb.set_trace()

#                formated_peaks_y = np.repeat(np.nan,70) 
#                formated_peaks_y[np.array(peaks_face[0][:,3],dtype=int)] = peaks_face[:,1]
                formated_peaks = np.hstack([formated_peaks_x,formated_peaks_y])
                
#                pdb.set_trace()
                return(formated_peaks)

#                peaks_hand = self.hand_model(oriImg_dat)
#                return(peaks_hand[0][:,[0,1,3]])

#                plt.scatter(peaks_hand[:,0],peaks_hand[:,1])
                break
            if case('face'):
#               To be implemented as a long list?               
#                pdb.set_trace()
                peaks_face = self.face_model(oriImg_dat)
                
                
                formated_peaks_x = np.repeat(np.nan,70)
                formated_peaks_y = np.repeat(np.nan,70)
                # for api:
                # only if face is detected:
                if len(peaks_face.shape)>0:

                    peaks_face = peaks_face[0]             
                
    #                formated_peaks_x = np.repeat(np.nan,70) 
                    
                    formated_peaks_x = peaks_face[:,0]
                    
    #                formated_peaks_y = np.repeat(np.nan,70) 
                    formated_peaks_y = peaks_face[:,1]
                
#                formated_peaks_x[np.array(peaks_face[0][:,3],dtype=int)] = peaks_face[:,0]
#                
#                formated_peaks_y = np.repeat(np.nan,70) 
#                formated_peaks_y[np.array(peaks_face[0][:,3],dtype=int)] = peaks_face[:,1]
                formated_peaks = np.hstack([formated_peaks_x,formated_peaks_y])
                
                
                return(formated_peaks)

#                plt.scatter(peaks_face[:,0],peaks_face[:,1])
                break
            if case('body_face'):
#                pdb.set_trace()
                peaks_body = self.body_model(oriImg_dat)
                
#                pdb.set_trace()
                formated_peaks_x = np.repeat(np.nan,26)
                formated_peaks_y = np.repeat(np.nan,26)
                
#                try:
#                    len(peaks_body[0])
#                except:
#                        pdb.set_trace()
                # check if there are peaks to work with:
#                pdb.set_trace()
                if len(peaks_body.shape)>0:
                    # a very crude way to deal with this right now. 
                    #  >26 are because it seems the algorithm has found multiple people
                    if peaks_body[0].shape[0]>26:
                        formated_peaks_x = peaks_body[0][0:26,0]                    
                        formated_peaks_y = peaks_body[0][0:26,1]
                    else:
#                        pdb.set_trace()
                        formated_peaks_x = peaks_body[0][:,0]                    
                        formated_peaks_y = peaks_body[0][:,1]
                # not for api
#                # check if there are peaks to work with:
#                if len(peaks_body[0])>0:
#                    # a very crude way to deal with this right now. 
#                    #  >26 are because it seems the algorithm has found multiple people
#                    if peaks_body[0].shape[0]>26:
#                        formated_peaks_x[np.array(peaks_body[0][0:26,3],dtype=int)] = peaks_body[0][0:26,0]                    
#                        formated_peaks_y[np.array(peaks_body[0][0:26,3],dtype=int)] = peaks_body[0][0:26,1]
#                    else:
##                        pdb.set_trace()
#                        formated_peaks_x[np.array(peaks_body[0][:,3],dtype=int)] = peaks_body[0][:,0]                    
#                        formated_peaks_y[np.array(peaks_body[0][:,3],dtype=int)] = peaks_body[0][:,1]
                formated_peaks_body = np.hstack([formated_peaks_x,formated_peaks_y])
                
                peaks_face = self.face_model(oriImg_dat)
                
                
                formated_peaks_x = np.repeat(np.nan,70)
                formated_peaks_y = np.repeat(np.nan,70)
                # for api:
                # only if face is detected:
                if len(peaks_face.shape)>0:

                    peaks_face = peaks_face[0]             
                
    #                formated_peaks_x = np.repeat(np.nan,70) 
                    
                    formated_peaks_x = peaks_face[:,0]
                    
    #                formated_peaks_y = np.repeat(np.nan,70) 
                    formated_peaks_y = peaks_face[:,1]
                
#                formated_peaks_x[np.array(peaks_face[0][:,3],dtype=int)] = peaks_face[:,0]
#                
#                formated_peaks_y = np.repeat(np.nan,70) 
#                formated_peaks_y[np.array(peaks_face[0][:,3],dtype=int)] = peaks_face[:,1]
                formated_peaks_face = np.hstack([formated_peaks_x,formated_peaks_y])  
                formated_peaks = np.hstack([formated_peaks_body,formated_peaks_face])
                return(formated_peaks)
            if case('all'):
                peaks_body,subset = self.body_model(oriImg)
                peaks_hand = self.hand_model(oriImg)
                peaks_face = self.face_model(oriImg)
                
#               To be implemented as a long list?               

                break
            if case():
                print("wrong model chosen")
                break   

    def predict_video(self,videoname):
        cap = cv2.VideoCapture(videoname)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #cv2.imshow('Frame',frame)
                self.predict_frame(frame)
            else:
                break