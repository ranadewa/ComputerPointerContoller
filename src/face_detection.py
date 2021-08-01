'''
Inherits from the Model class to specialse for face detection
'''
import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore 

class Face_Detection():
    '''
    Class for the Face Detection Model.
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
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        return self.exec_network.infer({self.input_blob: image})

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        height = 384
        width = 672
        image = cv2.resize(image, (width, height))
        image = image.transpose((2,0,1))
        image = np.expand_dims(image, axis=0)
        return image

    def preprocess_output(self, outputs, height, width):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        bounding_boxes = outputs[self.output_blob][0,0,:,:]
        scaled_boxes =  []

        for box in bounding_boxes:
            if(box[2] > 0.5):
                scaled_boxes.append(((int(box[3] * width), int(box[4] * height)),
                 (int(box[5] * width), int(box[6] * height))))

        
        return scaled_boxes


