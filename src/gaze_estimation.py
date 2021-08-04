import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore 

class Gaze_Estimation():
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        print(model_name)
        self.model_name = model_name
        self.model_bin = os.path.splitext(model_name)[0] + ".bin"
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if self.extensions and "CPU" in self.device:
            self.plugin.add_extension(self.extensions, self.device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=self.model_name, weights=self.model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, self.device)

        # Get the input layer
        self.input_blob = ((self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))


    def predict(self, input):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        return self.exec_network.infer(input)

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, left, right, head_pose):
        
        left, right = self.preprocess_eyes(left, right)

        head_pose_anlges = np.zeros((1,3))
        for i, v in enumerate(head_pose):
            head_pose_anlges[0, i] = v

        input = { 'head_pose_angles' : head_pose_anlges,
                  'left_eye_image': left,
                  'right_eye_image': right }

        return input

    def preprocess_eyes(self, left, right):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        height = 60
        width = 60

        left = cv2.resize(left, (width, height))
        left = left.transpose((2,0,1))
        left = np.expand_dims(left, axis=0)

        right = cv2.resize(right, (width, height))
        right = right.transpose((2,0,1))
        right = np.expand_dims(right, axis=0)

        return left, right

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        out = outputs[self.output_blob][0,:]

        return out



