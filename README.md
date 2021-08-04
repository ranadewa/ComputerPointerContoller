# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

### Models and IO
|Modle | Input, _format B, C, H, W & BGR color order_ | Output
|---|---|---|
| [Face detection](https://docs.openvinotoolkit.org/latest/omz_models_model_face_detection_adas_0001.html)  | Image, name: input, shape: 1, 3, 384, 672  |The net outputs blob with shape: 1, 1, 200, 7 in the format 1, 1, N, 7, where N is the numgber of detected bounding boxes. The results are sorted by confidence in decreasing order. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max] |
| [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_model_head_pose_estimation_adas_0001.html)  | Image, name: data, shape: 1, 3, 60, 60  |Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitch or roll)</br> name: angle_y_fc, shape: 1, 1 - Estimated yaw (in degrees).</br> name: angle_p_fc, shape: 1, 1 - Estimated pitch (in degrees).</br> name: angle_r_fc, shape: 1, 1 - Estimated roll (in degrees).|
| [Facial Lanmark Detection](https://docs.openvinotoolkit.org/latest/omz_models_model_landmarks_regression_retail_0009.html)| Image, name: data, shape: 1, 3, 48, 48| The net outputs a blob with the shape: 1, 10, containing a row-vector of 10 floating point values for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x4, y4). All the coordinates are normalized to be in range [0, 1]|
| [Gase Estimation](https://docs.openvinotoolkit.org/latest/omz_models_model_gaze_estimation_adas_0002.html) | Blob, name: left_eye_image, shape: 1, 3, 60, 60 </br> Blob, name: right_eye_image, shape: 1, 3, 60, 60 </br> Blob, name: head_pose_angles, shape: 1, 3 in the format B, C | The net output is a blob with name gaze_vector and the shape: 1, 3, containing Cartesian coordinates of gaze direction vector. Please note that the output vector is not normalizes and has non-unit length.|
#### Downloading the models
```
python D:\Programs\openvino_2021.1.110\deployment_tools\tools\model_downloader\downloader.py --name head-pose-estimation-adas-0001
python D:\Programs\openvino_2021.1.110\deployment_tools\tools\model_downloader\downloader.py --name landmarks-regression-retail-0009
python D:\Programs\openvino_2021.1.110\deployment_tools\tools\model_downloader\downloader.py --name gaze-estimation-adas-0002
python D:\Programs\openvino_2021.1.110\deployment_tools\tools\model_downloader\downloader.py --name face-detection-adas-0001
```
## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
