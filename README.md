# HAND KEYPOINT DETECTION

![](skeleton.gif)

This is a repo to execute live hand keypoint generation using OpenCV as implemented by https://www.learnopencv.com/hand-keypoint-detection-using-deep-learning-and-opencv/

I have added the model in the hand directory. The model can be downloaded from https://www.kaggle.com/changethetuneman/openpose-model?select=pose_iter_102000.caffemodel named as 'pose_iter_102000.caffemodel'  and save it in the hand directory.

The input video is saved as osy_test.mp4 and can be used by uncommenting the input_source variable.
The index for input cam is set as 0 by default and can be changes at the time of calling or in the '__init__' function if the class
The result has been stored in output.avi file

