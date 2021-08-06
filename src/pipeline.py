import argparse
import cv2
import logging
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import  Face_Detection
from head_pose_estimation import Head_Pose_Estimation
from facial_landmarks_detection import Facial_Landmarks_Detection
from gaze_estimation import Gaze_Estimation
from debug_utils import draw_axes, draw_gaze_vector


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    device = "The device name, if not 'CPU'"
    extension = "CPU extension if any"
    file = "Input file"
    setDebug = "set debug"


    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-i", help=file, default='../bin/demo.mp4')
    optional.add_argument("-d", help=device, default='CPU')
    optional.add_argument("-e", help=extension)
    optional.add_argument("-s", help=setDebug, default=0)
    args = parser.parse_args()

    return args

def get_face_coordinates(image, width, height):
    processed_image = fd_model.preprocess_input(image)
    fdm_out = fd_model.predict(processed_image)
    fdm_preprocessed_out = fd_model.preprocess_output(fdm_out, height, width)

    if(len(fdm_preprocessed_out) > 0):
        return fdm_preprocessed_out[0]
    else:
        None

def get_head_pose_estimate(cropped_face):
    hpe_preprocessed_in = hpe_model.preprocess_input(cropped_face)
    hpe_out = hpe_model.predict(hpe_preprocessed_in)
    return hpe_model.preprocess_output(hpe_out)

def get_facial_land_marks(cropped_face, width, height):
    fld_preprocessed_in = fld_model.preprocess_input(cropped_face)
    fld_out = fld_model.predict(fld_preprocessed_in)
    return  fld_model.preprocess_output(fld_out, width, height)

def extract_eyes(land_marks, face, offset):
    left_eye_min = (int(land_marks[0]) - offset, int(land_marks[1]) - offset)
    left_eye_max = (int(land_marks[0]) + offset, int(land_marks[1]) + offset)

    right_eye_min = (int(land_marks[2]) - offset, int(land_marks[3]) - offset)
    right_eye_max = (int(land_marks[2]) + offset, int(land_marks[3]) + offset)


    left_eye = face[left_eye_min[1]: left_eye_max[1], left_eye_min[0]: left_eye_max[0]]
    right_eye = face[right_eye_min[1]: right_eye_max[1], right_eye_min[0]: right_eye_max[0]]

    return left_eye, right_eye

def get_gaze_vector(left_eye, right_eye, yaw_pitch_roll):
    ge_preproccessed_in = ge_model.preprocess_input(left_eye, right_eye, yaw_pitch_roll)
    ge_out  = ge_model.predict(ge_preproccessed_in)
    ge_preprocessed_out = ge_model.preprocess_output(ge_out)
    return ge_preprocessed_out[:2]

def track_gaze(input, debug):

    if input == 'CAM':
        feed = InputFeeder(input_type='cam')
    else:
        feed=InputFeeder(input_type='video', input_file=input)
    feed.load_data()
    (width, height) = feed.dimensions()
    fps = feed.fps()
    logging.info('Video stream info: width: {}, height: {}, fps: {}'.format(width, height, fps))

    mouse_controller = MouseController('low', 'fast')
    
    if(debug):
        CODEC = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('debug_out.avi', CODEC, fps, (width,height))

    for batch in feed.next_batch():
        if(batch is None):
            logging.info('Video stream completed.')
            break
        
        face_coordinates = get_face_coordinates(batch, width, height)

        if(face_coordinates != None):
            (x_min,y_min), (x_max, y_max) = face_coordinates
            cropped_face = batch[y_min:y_max, x_min: x_max]

            (yaw, pitch, roll) = get_head_pose_estimate(cropped_face)

            fld_preprocessed_out = get_facial_land_marks(cropped_face, (x_max - x_min), (y_max - y_min))

            off_set = 30
            left_eye, right_eye = extract_eyes(fld_preprocessed_out, cropped_face, off_set)

            if(debug):
                center_of_face = (x_min + cropped_face.shape[1]/2, y_min  + cropped_face.shape[0]/2, 0)
                batch = draw_axes(batch, center_of_face, yaw, pitch, roll, 50, 950)

            expected_shape = (off_set * 2, off_set * 2, 3)
            if(left_eye.shape == expected_shape and right_eye.shape == expected_shape):

                x, y = get_gaze_vector(left_eye, right_eye, (yaw, pitch, roll))

                if(not(debug)):
                    mouse_controller.move(x, y)

            if(debug):
                batch = draw_gaze_vector(fld_preprocessed_out, (x_min, y_min), (x,y), batch)

        if(debug):
            out.write(batch)

    feed.close()
    if(debug):
        out.release()

fd_model = None
hpe_model = None 
fld_model = None
ge_model = None

def main():
    args = get_args()
    device = args.d
    extension = args.e

    global fd_model
    global hpe_model
    global fld_model
    global ge_model

    face_detection_model = '../models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml'
    fd_model = Face_Detection(model_name=face_detection_model, device = device, extensions=extension)
    fd_model.load_model()

    head_pose_estimation_model = '../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml'
    hpe_model = Head_Pose_Estimation(model_name=head_pose_estimation_model, device=device, extensions=extension)
    hpe_model.load_model()

    facial_landmark_detection_model = '../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml'
    fld_model = Facial_Landmarks_Detection(model_name=facial_landmark_detection_model, device=device, extensions=extension)
    fld_model.load_model()

    gaze_estimation_model = '../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml'
    ge_model = Gaze_Estimation(model_name=gaze_estimation_model, device=device, extensions=extension)
    ge_model.load_model()

    logging.basicConfig(filename='debug.log', level=logging.DEBUG)
    logging.info('Arguments: device: {}, extension: {}, file: {}, setDebug:{}'.format(device, extension, args.i, args.s))

    setDebug = False
    if(args.s == '1'):
        setDebug = True

    track_gaze(args.i, setDebug)


if __name__ == "__main__":
    main()
    