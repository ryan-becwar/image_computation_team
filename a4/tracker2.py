# adapted from https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

import cv2
import sys
import numpy as np
import time
from tracker import *

def quickNorm(img):
    return cv2.normalize(img, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64FC1)

if __name__ == '__main__':
    gaussianDim = 32
    gaussianSigma = 1.5
    sumFilters = None
    avgFilter = None
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    ok, frame = video.read()
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
        w = int(bbox[2])
        h = int(bbox[3])
        while True:
            ok, frame = video.read()
            if not ok: # video finished
                isDone = True
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if useASEF:
                _f = getLittleF(gray)
                #cv2.imshow('littlef', _f)
                #F = np.fft.fft2(_f)
                #F = cv2.dft(_f, None, cv2.DFT_COMPLEX_OUTPUT)
                #H = np.fft.fft2(avgFilter)
                #H = cv2.dft(avgFilter, None, cv2.DFT_COMPLEX_OUTPUT)
                #G = F * H
                #G = cv2.mulSpectrums(F, H, cv2.DFT_COMPLEX_OUTPUT)
                #_g = np.fft.ifft2(G)
                #_g = cv2.idft(G, None, cv2.DFT_REAL_OUTPUT)
                #cv2.imshow('littleg', cv2.normalize(_g, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64FC1))
                _f = quickNorm(_f)
                avgFilter = quickNorm(avgFilter)
                _g = cv2.filter2D(_f, cv2.CV_64FC1, avgFilter, borderType=cv2.BORDER_WRAP)
                _g = quickNorm(_g)
                cv2.imshow('littleg', _g)
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
            master = getMaster(gray, groundTruth, exactFilter, avgFilter)
            cv2.imshow('master', master)
            cv2.moveWindow('master', master.shape[1] + 10, 0)
            k = cv2.waitKey(1) & 0xff
            if k == ord('a'): # space to toggle ASEF/MOSSE tracking
                useASEF = not useASEF
            if k == ord('q'): # esc to quit
                isDone = True
                break

    video.release()
