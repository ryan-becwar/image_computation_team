# adapted from https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

import cv2
import sys
import numpy as np
import time
from tracker import *

def quickNorm(img):
    return cv2.normalize(img, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64FC1)

def updateSmoothing(x):
    global SMOOTHING_FACTOR
    SMOOTHING_FACTOR = x/10

def updateSigma(x):
    global gaussianSigma
    gaussianSigma = x

gaussianDim = 64
gaussianSigma = 2

if __name__ == '__main__':
    sumFilters = None
    avgFilter = None
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    print("Warming up webcam...")
    for i in range(0, 60):
        ok, frame = video.read()
    cv2.flip(frame, 1, frame)
    if not ok:
        print('Cannot read video file')
        sys.exit()
    isDone = False
    useASEF = False
    _g = None
    _f = None
    cv2.namedWindow('interface', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('sigma', 'interface', gaussianSigma, 60, updateSigma)
    cv2.createTrackbar('smoothing', 'interface', int(10 * SMOOTHING_FACTOR), 10, updateSmoothing)
    startTime = 0
    while not isDone:
        tracker = cv2.TrackerKCF_create()
        bbox = cv2.selectROI('Initialize KCF Tracker', frame, True, True)
        cv2.destroyWindow('Initialize KCF Tracker')
        ok = tracker.init(frame, bbox)
        w = int(bbox[2])
        h = int(bbox[3])
        while True:
            ok, frame = video.read()
            if not ok: # video finished
                isDone = True
                break
            cv2.flip(frame, 1, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if useASEF:
                _f = getLittleF(gray)
                avgFilter = quickNorm(avgFilter)

                _g = cv2.filter2D(_f, cv2.CV_64FC1, avgFilter, borderType=cv2.BORDER_WRAP)
                y, x = np.unravel_index(np.argmax(_g), _g.shape)
                groundTruth = getGroundTruth(x, y, gray.shape, gaussianDim, gaussianSigma)
                exactFilter = getExactFilter(gray, groundTruth)
                avgFilter = getExponentialFilter(exactFilter, avgFilter)
                w2 = int(w/2)
                h2 = int(h/2)
                p1 = (x - w2, y - h2)
                p2 = (x + w2, y + h2)
            else:
                ok, bbox = tracker.update(frame)
                if not ok: # tracker failed, get user input
                    break
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + w), int(bbox[1] + h))
                x, y = p1[0] + int(w/2), p1[1] + int(h/2)
                groundTruth = getGroundTruth(x, y, gray.shape, gaussianDim, gaussianSigma)
                exactFilter = getExactFilter(gray, groundTruth)
                avgFilter = getExponentialFilter(exactFilter, avgFilter)
            cv2.rectangle(gray, p1, p2, (255, 0, 0), 2, 1)
            master = getMaster(gray, groundTruth, exactFilter, avgFilter, _g, _f)
            fps = 'FPS: ' + ('%d' % (1 / (time.time() - startTime)))
            cv2.putText(master, fps, (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
            cv2.imshow('master', master)
            startTime = time.time()
            k = cv2.waitKey(1) & 0xff
            if k == ord('a'): # space to toggle ASEF/MOSSE tracking
                useASEF = not useASEF
            if k == ord('q'): # esc to quit
                isDone = True
                break

    video.release()
