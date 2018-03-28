# adapted from https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

import cv2
import sys
import numpy as np
import time
from tracker import *


def midpoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)


if __name__ == '__main__':
    gaussianDim = 32
    gaussianSigma = 1.5
    sumFilters = None
    avgFilter = None
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    time.sleep(1)
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    isDone = False
    useASEF = False
    while not isDone:
        tracker = cv2.TrackerKCF_create()
        bbox = cv2.selectROI(frame, False)
        ok = tracker.init(frame, bbox)
        while True:
            ok, frame = video.read()
            if not ok: # video finished
                isDone = True
                break
            ok, bbox = tracker.update(frame)
            if not ok: # tracker failed, get user input
                break
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            x, y = midpoint(p1, p2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            groundTruth = getGroundTruth(x, y, gray.shape, gaussianDim, gaussianSigma)
            exactFilter = getExactFilter(gray, groundTruth)
            avgFilter = getExponentialFilter(exactFilter, avgFilter)
            cv2.rectangle(gray, p1, p2, (255, 0, 0), 2, 1)
            master = getMaster(gray, groundTruth, exactFilter, avgFilter)
            cv2.imshow('master', master)
            cv2.moveWindow('master', frame.shape[0] + 10, 0)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                isDone = True
                break
    video.release()
