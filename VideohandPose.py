import cv2
import time
import numpy as np


class HandPose():
    def __init__(self, model, nPoints, pose_pair, input_source=0):
        self.input_source = input_source
        self.protofile, self.weightfile = model
        self.nPoints = nPoints
        self.POSE_PAIRS = pose_pair
        self.threshold = 0.2

    def window_size(self, frame):
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth / frameHeight
        inHeight = 368
        inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
        return inHeight, inWidth

    def draw_skeleton(self, frame, output):
        points = []
        for i in range(nPoints):
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            if prob > self.threshold:
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        return frame

    def feature_detector(self):
        cap = cv2.VideoCapture(self.input_source)
        hasframe, frame = cap.read()
        inHeight, inWidth = self.window_size(frame)
        net = cv2.dnn.readNetFromCaffe(self.protofile, self.weightfile)
        k = 0
        vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                     (frame.shape[1], frame.shape[0]))
        while True:
            k = k + 1
            t = time.time()
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv2.waitKey()
                break
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()
            print("forward = {}".format(time.time() - t))

            frame = self.draw_skeleton(frame, output)

            print("Time Taken for frame = {}".format(time.time() - t))

            cv2.imshow('Output-Skeleton', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            print("total = {}".format(time.time() - t))
            vid_writer.write(frame)
        vid_writer.release()


if __name__ == '__main__':
    protoFile = "hand/pose_deploy.prototxt"
    weightsFile = "hand/pose_iter_102000.caffemodel"
    nPoints = 22
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    gen = HandPose((protoFile, weightsFile), nPoints, POSE_PAIRS)
    gen.feature_detector()
