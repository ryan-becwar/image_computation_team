# adapted from https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

import cv2
import sys
import numpy as np
import time
import pickle

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

# taken from https://stackoverflow.com/a/19201448/2782424
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def getFrame(cap):
    ok, frame = cap.read()
    if not ok:  # video finished
        return None
    return cv2.flip(frame, 1, frame)

def getName(frame, keypoints, descriptors, model):
    return "objname"

if __name__ == '__main__':
    sumFilters = None
    avgFilter = None
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video.set(cv2.CAP_PROP_FPS, 60)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    print("Warming up webcam...")
    for i in range(0, 30):
        ok, frame = video.read()
    cv2.flip(frame, 1, frame)
    if not ok:
        print('Cannot read video file')
        sys.exit()
    isDone = False
    isTraining = False
    startTime = 0
    #Initlize SURF
    surf = cv2.xfeatures2d.SIFT_create(64)
    try:
        model = load_obj('model')
    except:
        model = {}
        save_obj(model, 'model')
    model = {}
    while not isDone:
        currentObj = input('Object name: ')
        frame = getFrame(video)
        if frame is None:
            break
        winname = 'Initialize Tracker'
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 0, 0)
        bbox = cv2.selectROI(winname, frame, True, True)
        if not bbox or bbox[2] == 0 or bbox[3] == 0:
            continue
        cv2.destroyWindow(winname)
        tracker = cv2.TrackerMOSSE_create()
        ok = tracker.init(frame, bbox)
        while True:
            frame = getFrame(video)
            if frame is None:
                isDone = True
                break

            # Update kcf tracker
            ok, bbox = tracker.update(frame)
            if not ok: # tracker failed, get user input
                break
            # bbox = topleftX, topleftY, W, H
            x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x2, y2 = x1+w, y1+h
            x, y = x1 + int(w/2), y1 + int(h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2, 1)
            crop = frame[y1:y2, x1:x2]

            # Update SURF
            keypoints, descriptors = surf.detectAndCompute(crop, None)
            frame[y1:y2, x1:x2] = cv2.drawKeypoints(crop, keypoints, None,(0,0,255),4)

            # Put some text on the image (post tracking)
            if isTraining == True:
                text = 'training ' + currentObj
                # update model with exponential decay
                model[currentObj] = model[currentObj] * SMOOTHING_FACTOR + (1 - SMOOTHING_FACTOR) * descriptors
            else:
                currentObj = getName(crop, keypoints, descriptors, model)
                text = 'recognizing ' + currentObj
            cv2.putText(frame, text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
            cv2.imshow('out', frame)
            cv2.moveWindow('out', 0, 0)
            k = cv2.waitKey(1) & 0xff
            if k == ord('q'): # q to quit
                isDone = True
                break
            elif k == ord('b'): # b to draw new bounding box
                isDone = False
                break
            elif k == ord('n'): # n to name an object
                currentObj = input('Object name: ')
            elif k == ord('t'): # t to toggle training
                isTraining = not isTraining
    video.release()
