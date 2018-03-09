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

def convertToSpatial(fourier):
    spatial = np.fft.ifft2(fourier)
    shifted = np.fft.fftshift(np.abs(spatial) * 3000)
    rotated = cv2.rotate(shifted, cv2.ROTATE_180)
    return rotated

def getExactFilter(lap, groundTruth):
    fourierF = np.fft.fft2(lap)
    fourierG = np.fft.fft2(groundTruth)
    fourierH = fourierG / fourierF
    return fourierH
    #spatialH = np.fft.ifft2(fourierH)
    #shiftedH = np.fft.fftshift(np.abs(spatialH) * 3000)
    #rotatedH = cv2.rotate(shiftedH, cv2.ROTATE_180)
    #return rotatedH

def getLocation(lap, fourierH):
    fourierF = np.fft.fft2(lap)
    fourierG = fourierF * fourierH

    return fourierG


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



def getMaster(cropSize, frame, groundTruth, exactFilter, avgFilter):
    master = np.zeros((cropSize * 2, cropSize * 2))
    master[0:cropSize, 0:cropSize] = frame / 255
    master[0:cropSize, cropSize:cropSize * 2] = groundTruth
    master[cropSize:cropSize * 2, 0:cropSize] = exactFilter
    master[cropSize:cropSize * 2, cropSize:cropSize * 2] = avgFilter
    return master


if __name__ == '__main__':
    gaussianDim = 32
    gaussianSigma = 3
    threshold = 32
    xOff = 350
    yOff = 100
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
        lap = cv2.Laplacian(gray, cv2.CV_8U)
        exactFilterFourier = getExactFilter(lap, groundTruth)
        exactFilter = convertToSpatial(exactFilterFourier)

        #avgFilter, sumFilters = getAvgFilter(exactFilter, sumFilters, N)
        avgFilterFourier = getExponentialFilter(exactFilterFourier, avgFilter)
        avgFilter = convertToSpatial(avgFilterFourier)

        locationOutFourier = getLocation(lap, exactFilterFourier)
        locationOut = convertToSpatial(locationOutFourier)
        cv2.imshow('tracked', locationOut)


        master = getMaster(cropSize, gray, groundTruth, exactFilter, avgFilter)
        cv2.imshow('master', master)

        if cv2.waitKey(17) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
