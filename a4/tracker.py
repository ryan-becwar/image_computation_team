import cv2, sys
import numpy as np

SMOOTHING_FACTOR = 0.1

def stupidTrack(img, thresh):
    """given a grayscale image and threshold, find the center of the
    rectangle created to contain pixels below that threshold"""
    mask = np.argwhere(img < thresh)
    y1, x1 = mask[0]
    y2, x2 = mask[-1]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def getGaussian(rows, cols, sigma):
    return cv2.getGaussianKernel(rows, sigma, cv2.CV_64F) @ \
           np.transpose(cv2.getGaussianKernel(cols, sigma, cv2.CV_64F))


def getGroundTruth(x, y, dims, gaussianDim, gaussianSigma):
    halfGaussDim = int(gaussianDim / 2)
    normal = np.zeros(dims)
    kernel = getGaussian(gaussianDim, gaussianDim, gaussianSigma) * 255
    y = halfGaussDim if y < halfGaussDim else y
    y = dims[0] - halfGaussDim if y > dims[0] - halfGaussDim else y
    x = halfGaussDim if x < halfGaussDim else x
    x = dims[1] - halfGaussDim if x > dims[1] - halfGaussDim else x
    normal[y - halfGaussDim:y + halfGaussDim, x - halfGaussDim:x + halfGaussDim] = kernel
    return normal


def getLittleF(gray):
    sx = cv2.Sobel(gray, cv2.CV_8UC1, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_8UC1, 0, 1)
    s = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)
    s = cv2.normalize(s, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64FC1)
    hanning = cv2.createHanningWindow((s.shape[1], s.shape[0]), cv2.CV_64FC1) * s
    return hanning


def getExactFilter(gray, groundTruth):
    fourierF = np.fft.fft2(getLittleF(gray))
    fourierG = np.fft.fft2(groundTruth)
    fourierH = fourierG / fourierF
    spatialH = np.fft.ifft2(fourierH)
    shiftedH = np.fft.fftshift(np.abs(spatialH))
    rotatedH = cv2.normalize(cv2.rotate(shiftedH, cv2.ROTATE_180), None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64FC1)
    return rotatedH


def getAvgFilter(exactFilter, sumFilters, N):
    if sumFilters is None:
        sumFilters = np.copy(exactFilter)
    np.add(sumFilters, exactFilter, out=sumFilters)
    return sumFilters / N, sumFilters


def getExponentialFilter(exactFilter, sumFilters):
    if sumFilters is None:
        sumFilters = np.copy(exactFilter)
    np.add(SMOOTHING_FACTOR * exactFilter, (1 - SMOOTHING_FACTOR) * sumFilters, out=sumFilters)
    return cv2.normalize(sumFilters, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64FC1)


def quickNorm255(img):
    return cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64FC1)

def getMaster(frame, groundTruth, exactFilter, avgFilter, calcG=None, edges=None):
    h = frame.shape[0]
    w = frame.shape[1]
    calcG = np.zeros((h,w)) if calcG is None else calcG
    edges = np.zeros((h,w)) if edges is None else edges
    master = np.zeros((h * 2, w * 3))
    master[0:h, 0:w] = quickNorm255(frame)
    master[0:h, w:w * 2] = quickNorm255(groundTruth)
    master[0:h, w * 2:w * 3] = quickNorm255(calcG)
    master[h:h * 2, 0:w] = quickNorm255(exactFilter)
    master[h:h * 2, w:w * 2] = quickNorm255(avgFilter)
    master[h:h * 2, w * 2:w * 3] = quickNorm255(edges)

    return master


if __name__ == '__main__':
    gaussianDim = 32
    gaussianSigma = 3
    threshold = 32
    xOff = 300
    yOff = 0
    cropSize = 512
    sumFilters = None
    avgFilter = None
    cap = cv2.VideoCapture(sys.argv[1])
    N = 0
    if not cap.isOpened():
        cap.open()

    while cap.isOpened():
        N += 1
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[yOff:yOff + cropSize, xOff:xOff + cropSize]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y = stupidTrack(gray, threshold) #ground truth x and y coords
        groundTruth = getGroundTruth(x, y, gray.shape, gaussianDim, gaussianSigma)
        exactFilter = getExactFilter(gray, groundTruth)
        #avgFilter, sumFilters = getAvgFilter(exactFilter, sumFilters, N)
        avgFilter = getExponentialFilter(exactFilter, avgFilter)
        master = getMaster(gray, groundTruth, exactFilter, avgFilter)
        cv2.imshow('master', master)

        if cv2.waitKey(17) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
