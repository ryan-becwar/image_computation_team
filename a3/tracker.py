import cv2, sys
import numpy as np
import matplotlib.pyplot as plt


def stupidTrack(img, thresh):
    """given a grayscale image and threshold, find the center of the
    rectangle created to contain pixels below that threshold"""
    mask = np.argwhere(img < thresh)
    y1, x1 = mask[0]
    y2, x2 = mask[-1]
    return int((x1 + x2)/2), int((y1 + y2)/2)


def getGaussian(rows, cols, sigma):
    return cv2.getGaussianKernel(rows, sigma, cv2.CV_64F) @ \
           np.transpose(cv2.getGaussianKernel(cols, sigma, cv2.CV_64F))


def getGroundTruth(x, y, dims, gaussianDim, gaussianSigma):
    halfGaussDim = int(gaussianDim / 2)
    normal = np.zeros(dims)
    kernel = getGaussian(gaussianDim, gaussianDim, gaussianSigma) * 255
    normal[y - halfGaussDim:y + halfGaussDim, x - halfGaussDim:x + halfGaussDim] = kernel
    return normal


if __name__ == '__main__':
    gaussianDim = 32
    gaussianSigma = 3
    threshold = 32
    xOff = 350
    yOff = 100
    cropSize = 512
    cap = cv2.VideoCapture(sys.argv[1])
    if not cap.isOpened():
        cap.open()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[yOff:yOff+cropSize, xOff:xOff+cropSize]
        cv2.imshow('input', frame)
        x, y = stupidTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), threshold)
        groundTruth = getGroundTruth(x, y, frame.shape[:-1], gaussianDim, gaussianSigma)
        cv2.imshow('ground truth', groundTruth)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


