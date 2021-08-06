import os
import cv2
import numpy as np
from model import Model

class Facial_Landmarks_Detection(Model):
    '''
    Class for the Face Landmard detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        super().__init__(model_name, device, extensions)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        height = 48
        width = 48
        image = cv2.resize(image, (width, height))
        image = image.transpose((2,0,1))
        image = np.expand_dims(image, axis=0)
        return image

    def preprocess_output(self, outputs, width, height):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        out = outputs[self.output_blob][0,:,0,0]

        for index, _ in enumerate(out):
            if( index % 2 == 0):
                out[index] = out[index] * width
            else:
                out[index] = out[index] * height

        return out



