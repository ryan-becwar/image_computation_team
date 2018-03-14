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
    normal[y - halfGaussDim:y + halfGaussDim, x - halfGaussDim:x + halfGaussDim] = kernel
    return normal


def getExactFilter(gray, groundTruth):
    blur = cv2.GaussianBlur(gray, (3, 3), 0, dst=None, sigmaY=0)
    edge = cv2.Laplacian(blur, cv2.CV_8U)
    #cv2.imshow('blur', blur)
    #edgex = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    #edgey = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    #edge = np.abs(edgex)*0.1 + np.abs(edgey)*0.1
    cv2.imshow('edges', edge)
    fourierF = np.fft.fft2(edge)
    fourierG = np.fft.fft2(groundTruth)
    fourierH = fourierG / fourierF
    spatialH = np.fft.ifft2(fourierH)
    shiftedH = np.fft.fftshift(np.abs(spatialH) * 4000)
    rotatedH = cv2.rotate(shiftedH, cv2.ROTATE_180)
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
    return sumFilters


def getMaster(frame, groundTruth, exactFilter, avgFilter):
    h = frame.shape[0]
    w = frame.shape[1]
    master = np.zeros((h * 2, w * 2))
    master[0:h, 0:w] = frame / 255
    master[0:h, w:w * 2] = groundTruth
    master[h:h * 2, 0:w] = exactFilter
    master[h:h * 2, w:w * 2] = avgFilter
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
