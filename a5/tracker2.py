import cv2
import sys
import numpy as np
import time
import pickle

SMOOTHING_FACTOR = 0.1

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

def getName(matcher, descriptors, model):
    matches = [(n, matcher.knnMatch(d, descriptors, k=2)) for n, d in model.items()]

    masterMasks = []
    for name, modelDescriptors in matches:
        matchesMask = [[0, 0] for i in range(len(modelDescriptors))]
        for i,(m,n) in enumerate(modelDescriptors):  # ratio test as per Lowe's paper
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1, 0]
        masterMasks.append((name, matchesMask))
    return [(name, sum([x_i for x_i, y_i in x])) for name, x in masterMasks]


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
    isRecognizing = False
    startTime = 0
    #Initialize SURF
    detector = cv2.xfeatures2d.SURF_create(1000)
    #Initialize FLANN

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    name = ''
    ok = True
    try:
        model = load_obj('model')
    except:
        model = {}
        save_obj(model, 'model')
    while not isDone:
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
        if ok:  # don't ask again on tracker failures
            name = input("Object name: ")
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
            keypoints, descriptors = detector.detectAndCompute(crop, None)
            frame[y1:y2, x1:x2] = cv2.drawKeypoints(crop, keypoints, None,(0,0,255),4)

            # Put some text on the image (post tracking)
            if isTraining == True:
                text = 'training ' + name
                # update model with exponential decay
                if name in model:
                    #model[currentObj] = model[currentObj] * SMOOTHING_FACTOR + (1 - SMOOTHING_FACTOR) * descriptors
                    model[name] = descriptors
                else:
                    model[name] = descriptors
            elif isRecognizing == True:
                if model:
                    votesList = getName(matcher, descriptors, model)
                    text = 'recognizing ' + max(votesList, key=lambda x: x[1])[0]
                    votesText = str(votesList)
                    cv2.putText(frame, votesText, (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                else:
                    text = 'recognizing ???'
            else:
                text = ''
                help = '(b)ound; (t)rain; (r)ecognize; (s)ave; (l)oad; (q)uit'
            cv2.putText(frame, text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            cv2.putText(frame, help, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            cv2.imshow('out', frame)
            #cv2.imshow('matches', matchImage)
            cv2.moveWindow('out', 0, 0)
            k = cv2.waitKey(1) & 0xff
            if k == ord('q'): # q to quit
                isDone = True
                break
            elif k == ord('b'): # b to draw new bounding box and stop recognizing and training
                isTraining = False
                isRecognizing = False
                isDone = False
                cv2.destroyAllWindows()
                break
            elif k == ord('t'): # t to start training and stop recognizing
                isTraining = True
                isRecognizing = False
            elif k == ord('r'): # r to start recognizing and stop training
                isTraining = False
                isRecognizing = True
            elif k == ord('s'): # s to save to disk
                save_obj(model, 'model')
            elif k == ord('s'): # l to load from disk
                model = load_obj('model')
    video.release()
