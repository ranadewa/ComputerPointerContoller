import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore 
from model import Model

class Head_Pose_Estimation(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        super().__init__(model_name, device, extensions)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        height = 60
        width = 60
        image = cv2.resize(image, (width, height))
        image = image.transpose((2,0,1))
        image = np.expand_dims(image, axis=0)
        return image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        yaw = outputs['angle_y_fc'][0][0]
        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]
    
        return (yaw, pitch, roll)



