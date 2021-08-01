import argparse
import cv2
from input_feeder import InputFeeder
from face_detection import  Face_Detection

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    device = "The device name, if not 'CPU'"
    extension = "CPU extension if any"
    file = "Input file"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-i", help=file, default='../bin/demo.mp4')
    optional.add_argument("-d", help=device, default='CPU')
    optional.add_argument("-e", help=extension)
    args = parser.parse_args()

    return args

def track_gaze(args):
    device = args.d
    extension = args.e
    print('file: {}, device: {}'.format(args.i, args.d))

    feed=InputFeeder(input_type='video', input_file=args.i)
    feed.load_data()
    (width, height) = feed.dimensions()

    fd_model = '../models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml'
    face_detection_model = Face_Detection(model_name=fd_model, device = device, extensions=extension)
    face_detection_model.load_model()

    hpe_model = '../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml'

    CODEC = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('out.avi', CODEC, 30, (width,height))

    for batch in feed.next_batch():
        if(batch is None):
            break
        
        processed_image = face_detection_model.preprocess_input(batch)
        fdm_out = face_detection_model.predict(processed_image)
        fdm_preprocessed_out = face_detection_model.preprocess_output(fdm_out, height, width)

        if(len(fdm_preprocessed_out) > 0):
            bottom, top = fdm_preprocessed_out[0]
            cv2.rectangle(batch, bottom, top, (255,0,), 5)
            out.write(batch)

    feed.close()
    out.release()
    
def main():
    args = get_args()
    track_gaze(args)


if __name__ == "__main__":
    main()